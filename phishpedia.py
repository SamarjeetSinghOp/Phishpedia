import time
from datetime import datetime
import argparse
import os
import torch
import cv2
import psutil
import json
import numpy as np
from PIL import Image
from configs import load_config
from logo_recog import vis
from detector_adapter import pred_detector, config_rfdetr
from logo_matching import check_domain_brand_inconsistency
from tqdm import tqdm
import re
import subprocess
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_system_metrics():
    """Get current system resource usage metrics"""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Get process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_memory_mb = process_memory.rss / (1024**2)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': round(memory_used_gb, 2),
            'memory_total_gb': round(memory_total_gb, 2),
            'process_memory_mb': round(process_memory_mb, 2),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def get_image_info(image_path):
    """Get image metadata"""
    try:
        with Image.open(image_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format,
                'size_bytes': os.path.getsize(image_path),
                'resolution': f"{img.width}x{img.height}"
            }
    except Exception as e:
        return {'error': str(e)}

# ----------------- SIMPLIFIED VISUALIZATION FUNCTION -----------------
def save_detection_visualization(image_path, pred_boxes, pred_target, matched_coord, siamese_conf, output_dir):
    """
    Save clean visualization showing ONLY matched brand logos.
    - No black footer bar
    - No generic logo annotations
    - ONLY shows logos that successfully matched to a brand
    - Brand name displayed below the matched logo (no confidence scores)
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image from {image_path}")
            return None
            
        # Create copy for visualization
        vis_img = img.copy()
        height, width = vis_img.shape[:2]
        
        # ONLY draw if we have a successful brand match
        if matched_coord is not None and pred_target is not None:
            if len(matched_coord) >= 4:
                x1, y1, x2, y2 = map(int, matched_coord[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                # Draw red bounding box for matched brand logo
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Add brand name BELOW the bounding box
                brand_text = f'{pred_target}'
                text_size = cv2.getTextSize(brand_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Position text below the box
                text_x = x1
                text_y = y2 + text_size[1] + 10  # Below the box with some padding
                
                # Ensure text stays within image bounds
                if text_y + text_size[1] > height:
                    text_y = y1 - 10  # Move above if no space below
                    
                if text_x + text_size[0] > width:
                    text_x = width - text_size[0] - 10
                
                # Draw background rectangle for better text visibility
                cv2.rectangle(
                    vis_img,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 255),
                    -1
                )
                
                # Add brand name text in white
                cv2.putText(
                    vis_img, 
                    brand_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255), 
                    2
                )
        
        # Save the final image (clean, only matched brands shown)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'detection_visual.png')
        success = cv2.imwrite(output_path, vis_img)
        
        if success:
            if pred_target:
                print(f"✅ Visualization saved with brand match: {pred_target}")
            else:
                print(f"✅ Visualization saved (no brand matches to display)")
            return output_path
        else:
            print(f"❌ Failed to save visualization to {output_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        import traceback
        traceback.print_exc()
        return None
# ----------------- END OF SIMPLIFIED FUNCTION -----------------

def result_file_write(f, folder, url, phish_category, pred_target, matched_domain, siamese_conf, logo_recog_time,
                      logo_match_time):
    f.write(folder + "\t")
    f.write(url + "\t")
    f.write(str(phish_category) + "\t")
    f.write(str(pred_target) + "\t")  # write top1 prediction only
    f.write(str(matched_domain) + "\t")
    f.write(str(siamese_conf) + "\t")
    f.write(str(round(logo_recog_time, 4)) + "\t")
    f.write(str(round(logo_match_time, 4)) + "\n")


class PhishpediaWrapper:
    _caller_prefix = "PhishpediaWrapper"
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __init__(self, detector_override: str | None = None):
        self._detector_override = detector_override
        self._load_config(reload_targetlist=False)
        # expose last-run data for API/webtool access
        self.last_pred_boxes = None
        self.last_matched_coord = None
        self.last_pred_target = None
        # siamese overrides (env-controlled)
        self._siamese_threshold_override = None
        try:
            _th = os.environ.get('PHISHPEDIA_SIAMESE_THRE')
            if _th is not None and _th.strip() != '':
                self._siamese_threshold_override = float(_th)
        except Exception:
            self._siamese_threshold_override = None
        # enable resolution alignment by default (improves recall)
        self._siamese_align = os.environ.get('PHISHPEDIA_SIAMESE_ALIGN', 'true').lower() in ('1','true','yes')
        # aspect ratio check can filter some FPs; default off for recall
        self._siamese_archeck = os.environ.get('PHISHPEDIA_SIAMESE_ARCHECK', 'false').lower() in ('1','true','yes')

    def _load_config(self, reload_targetlist: bool = False):
        self.ELE_MODEL, self.SIAMESE_THRE, self.SIAMESE_MODEL, \
            self.LOGO_FEATS, self.LOGO_FILES, \
            self.DOMAIN_MAP_PATH = load_config(reload_targetlist=reload_targetlist,
                                               detector_override=self._detector_override)
        print(f'Length of reference list = {len(self.LOGO_FEATS)}')

    def reload(self, detector_override: str | None = None, reload_targetlist: bool = False):
        if detector_override is not None:
            self._detector_override = detector_override
        self._load_config(reload_targetlist=reload_targetlist)
        # reset caches
        self.last_pred_boxes = None
        self.last_matched_coord = None
        self.last_pred_target = None

    def detect_logo_brand_only(self, screenshot_path, siamese_threshold: float | None = None,
                              align: bool | None = None, archeck: bool | None = None):
        """
        Simplified method for logo detection and brand identification only.
        No phishing analysis, no extensive logging - just fast logo + brand detection.
        Returns: pred_target, pred_boxes, siamese_conf, logo_recog_time, logo_match_time
        """
        pred_target = None
        siamese_conf = None
        logo_match_time = 0.0

        # Step 1: Logo Detection
        rcnn_start_time = time.time()
        try:
            pred_boxes = pred_detector(im_path=screenshot_path, detector=self.ELE_MODEL)
            logo_recog_time = time.time() - rcnn_start_time
        except Exception as e:
            logo_recog_time = time.time() - rcnn_start_time
            print(f"Detector Error: {str(e)}")
            return None, None, None, logo_recog_time, 0.0

        # Early exit if no logos
        if pred_boxes is None or len(pred_boxes) == 0:
            return None, pred_boxes, None, logo_recog_time, 0.0

        # Step 2: Siamese Brand Matching
        siamese_start_time = time.time()
        try:
            # Apply thresholds
            _th = (siamese_threshold if siamese_threshold is not None else
                   (self._siamese_threshold_override if self._siamese_threshold_override is not None else self.SIAMESE_THRE))
            _align = self._siamese_align if align is None else bool(align)
            _archeck = self._siamese_archeck if archeck is None else bool(archeck)
            
            pred_target, _, _, siamese_conf = check_domain_brand_inconsistency(
                logo_boxes=pred_boxes,
                domain_map_path=self.DOMAIN_MAP_PATH,
                model=self.SIAMESE_MODEL,
                logo_feat_list=self.LOGO_FEATS,
                file_name_list=self.LOGO_FILES,
                url="",  # Not needed for brand detection only
                shot_path=screenshot_path,
                similarity_threshold=_th,
                topk=1,
                grayscale=False,
                do_resolution_alignment=_align,
                do_aspect_ratio_check=_archeck)
            logo_match_time = time.time() - siamese_start_time
        except Exception as e:
            logo_match_time = time.time() - siamese_start_time
            print(f"Siamese Error: {str(e)}")
            pred_target = None

        return pred_target, pred_boxes, siamese_conf, logo_recog_time, logo_match_time

    def test_orig_phishpedia(self, url, screenshot_path, html_path, analysis_dir=None,
                             siamese_threshold: float | None = None,
                             align: bool | None = None,
                             archeck: bool | None = None):
        # 0 for benign, 1 for phish, default is benign
        phish_category = 0
        pred_target = None
        matched_domain = None
        siamese_conf = None
        plotvis = None
        logo_match_time = 0.0
        matched_coord = None

        # Initialize analysis data
        analysis_data = {
            'folder': os.path.basename(os.path.dirname(screenshot_path)),
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'image_info': get_image_info(screenshot_path),
            'system_metrics': {},
            'model_performance': {},
            'detection_results': {},
            'error_logs': []
        }

        print(f"Entering phishpedia for: {analysis_data['folder']}")

        # Get initial system metrics
        analysis_data['system_metrics']['start'] = get_system_metrics()

        # Step 1: Detector
        print("🔍 Starting Logo Detection...")
        rcnn_start_time = time.time()
        pred_boxes = None
        try:
            pred_boxes = pred_detector(im_path=screenshot_path, detector=self.ELE_MODEL)
            logo_recog_time = time.time() - rcnn_start_time
            analysis_data['model_performance']['detector'] = {
                'processing_time': logo_recog_time,
                'success': True,
                'num_detections': int(len(pred_boxes)) if pred_boxes is not None else 0
            }
            if pred_boxes is not None and hasattr(pred_boxes, 'detach'):
                analysis_data['detection_results']['logo_boxes'] = pred_boxes.detach().cpu().numpy().tolist()
            elif pred_boxes is not None:
                analysis_data['detection_results']['logo_boxes'] = np.asarray(pred_boxes).tolist()
            else:
                analysis_data['detection_results']['logo_boxes'] = []
            print(f"✅ Detector found {analysis_data['model_performance']['detector']['num_detections']} logo regions")
        except Exception as e:
            logo_recog_time = time.time() - rcnn_start_time
            analysis_data['model_performance']['detector'] = {
                'processing_time': logo_recog_time,
                'success': False,
                'error': str(e)
            }
            analysis_data['error_logs'].append(f"Detector Error: {str(e)}")
            pred_boxes = None

        # After detector metrics
        analysis_data['system_metrics']['after_detector'] = get_system_metrics()

        # Visualization of raw boxes
        plotvis = vis(screenshot_path, pred_boxes)

        # Early exit if no logos
        if pred_boxes is None or len(pred_boxes) == 0:
            print('No logo is detected - returning as benign')
            analysis_data['detection_results']['final_result'] = 'benign_no_logos'
            if analysis_dir:
                self._save_analysis_data(analysis_data, analysis_dir)
                save_detection_visualization(screenshot_path, pred_boxes, None, None, None, analysis_dir)
            return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time

        # Step 2: Siamese matching
        print("🧠 Starting Siamese Logo Matching...")
        siamese_start_time = time.time()
        try:
            # apply overrides or fallbacks
            _th = (siamese_threshold if siamese_threshold is not None else
                   (self._siamese_threshold_override if self._siamese_threshold_override is not None else self.SIAMESE_THRE))
            _align = self._siamese_align if align is None else bool(align)
            _archeck = self._siamese_archeck if archeck is None else bool(archeck)
            pred_target, matched_domain, matched_coord, siamese_conf = check_domain_brand_inconsistency(
                logo_boxes=pred_boxes,
                domain_map_path=self.DOMAIN_MAP_PATH,
                model=self.SIAMESE_MODEL,
                logo_feat_list=self.LOGO_FEATS,
                file_name_list=self.LOGO_FILES,
                url=url,
                shot_path=screenshot_path,
                similarity_threshold=_th,
                topk=1,
                grayscale=False,
                do_resolution_alignment=_align,
                do_aspect_ratio_check=_archeck)
            logo_match_time = time.time() - siamese_start_time
            analysis_data['model_performance']['siamese'] = {
                'processing_time': logo_match_time,
                'success': True,
                'matched_brand': pred_target,
                'confidence': float(siamese_conf) if siamese_conf else 0.0,
                'matched_domains': matched_domain
            }
            if pred_target:
                print(f"✅ Siamese matched: {pred_target}")
            else:
                print("❌ Siamese found no brand matches")
        except Exception as e:
            logo_match_time = time.time() - siamese_start_time
            analysis_data['model_performance']['siamese'] = {
                'processing_time': logo_match_time,
                'success': False,
                'error': str(e)
            }
            analysis_data['error_logs'].append(f"Siamese Error: {str(e)}")
            pred_target = None

        # Final system metrics
        analysis_data['system_metrics']['final'] = get_system_metrics()

        # Cache last-run details
        try:
            self.last_pred_boxes = pred_boxes.detach().cpu().numpy() if hasattr(pred_boxes, 'detach') else pred_boxes
        except Exception:
            self.last_pred_boxes = pred_boxes
        self.last_matched_coord = matched_coord
        self.last_pred_target = pred_target

        # Decision
        if pred_target is None:
            print('Did not match to any brand, report as benign')
            analysis_data['detection_results']['final_result'] = 'benign_no_match'
        else:
            print('Match to Target: {}'.format(pred_target))
            phish_category = 1
            analysis_data['detection_results']['final_result'] = 'phishing_detected'

        # Final analysis data
        analysis_data['detection_results'].update({
            'phish_category': phish_category,
            'predicted_target': pred_target,
            'matched_domain': matched_domain,
            'siamese_confidence': float(siamese_conf) if siamese_conf else 0.0,
            'total_processing_time': logo_recog_time + logo_match_time
        })

        # Save comprehensive analysis
        if analysis_dir:
            self._save_analysis_data(analysis_data, analysis_dir)
            save_detection_visualization(screenshot_path, pred_boxes, pred_target, matched_coord, siamese_conf, analysis_dir)

        return phish_category, pred_target, matched_domain, plotvis, siamese_conf, pred_boxes, logo_recog_time, logo_match_time

    def _save_analysis_data(self, analysis_data, analysis_dir):
        """Save comprehensive analysis data to JSON"""
        try:
            os.makedirs(analysis_dir, exist_ok=True)
            analysis_file = os.path.join(analysis_dir, 'analysis.json')
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                return obj
            
            # Clean data for JSON
            clean_data = json.loads(json.dumps(analysis_data, default=convert_for_json))
            
            with open(analysis_file, 'w') as f:
                json.dump(clean_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save analysis data: {e}")


if __name__ == '__main__':

    '''run'''
    today = datetime.now().strftime('%Y%m%d')

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=str)
    parser.add_argument("--output_txt", default=f'{today}_results.txt', help="Output txt path")
    parser.add_argument("--analysis_mode", action='store_true', help="Enable detailed analysis mode")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--detector", choices=['rcnn','rfdetr'], help="Override detector type (else from configs.yaml)")
    parser.add_argument("--export_pip_list", action='store_true', help="Export pip list at end of run")
    args = parser.parse_args()

    request_dir = args.folder
    phishpedia_cls = PhishpediaWrapper(detector_override=args.detector)
    result_txt = args.output_txt
    analysis_mode = args.analysis_mode

    os.makedirs(request_dir, exist_ok=True)
    
    # Initialize analysis variables
    analysis_base_dir = None
    summary_data = None
    
    # Create analysis directory if in analysis mode
    if analysis_mode:
        analysis_base_dir = f"analysis_{today}"
        os.makedirs(analysis_base_dir, exist_ok=True)
        print(f"📊 Analysis mode enabled. Results will be saved to: {analysis_base_dir}")
        
        # Create summary file for all analyses
        summary_data = {
            'start_time': datetime.now().isoformat(),
            'total_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'performance_summary': {
                'avg_rcnn_time': 0,
                'avg_siamese_time': 0,
                'avg_total_time': 0,
                'avg_cpu_usage': 0,
                'avg_memory_usage': 0
            },
            'detection_summary': {
                'total_logos_detected': 0,
                'total_brands_matched': 0,
                'phishing_detected': 0,
                'benign_results': 0
            }
        }

    # Get list of folders and optionally limit for testing
    folders = [f for f in os.listdir(request_dir) 
             if os.path.isdir(os.path.join(request_dir, f)) 
             and not f.startswith('.')]  # Filter out system files like .DS_Store
    if args.max_samples:
        folders = folders[:args.max_samples]
        print(f"🔢 Limited to {args.max_samples} samples for testing")

    # Performance tracking
    all_rcnn_times = []
    all_siamese_times = []
    all_cpu_usage = []
    all_memory_usage = []

    for folder in tqdm(folders, desc="Processing websites"):
        html_path = os.path.join(request_dir, folder, "html.txt")
        screenshot_path = os.path.join(request_dir, folder, "shot.png")
        info_path = os.path.join(request_dir, folder, 'info.txt')

        if not os.path.exists(screenshot_path):
            continue
        if not os.path.exists(html_path):
            html_path = os.path.join(request_dir, folder, "index.html")
        if not os.path.exists(info_path):
            print(f"Warning: info.txt not found for {folder}. Skipping.")
            continue

        with open(info_path, 'r') as file:
            url = file.read().strip()
        
        # Skip duplicate check in analysis mode to allow re-processing
        if not analysis_mode and os.path.exists(result_txt):
            with open(result_txt, 'r', encoding='ISO-8859-1') as file:
                if url in file.read():
                    continue

        _forbidden_suffixes = r"\.(mp3|wav|wma|ogg|mkv|zip|tar|xz|rar|z|deb|bin|iso|csv|tsv|dat|txt|css|log|xml|sql|mdb|apk|bat|exe|jar|wsf|fnt|fon|otf|ttf|ai|bmp|gif|ico|jp(e)?g|png|ps|psd|svg|tif|tiff|cer|rss|key|odp|pps|ppt|pptx|c|class|cpp|cs|h|java|sh|swift|vb|odf|xlr|xls|xlsx|bak|cab|cfg|cpl|cur|dll|dmp|drv|icns|ini|lnk|msi|sys|tmp|3g2|3gp|avi|flv|h264|m4v|mov|mp4|mp(e)?g|rm|swf|vob|wmv|doc(x)?|odt|rtf|tex|wks|wps|wpd)$"
        if re.search(_forbidden_suffixes, url, re.IGNORECASE):
            continue

        # Create analysis directory for this sample
        sample_analysis_dir = None
        if analysis_mode:
            sample_analysis_dir = os.path.join(analysis_base_dir, folder)
            os.makedirs(sample_analysis_dir, exist_ok=True)

        try:
            phish_category, pred_target, matched_domain, \
                plotvis, siamese_conf, pred_boxes, \
                logo_recog_time, logo_match_time = phishpedia_cls.test_orig_phishpedia(
                    url, screenshot_path, html_path, sample_analysis_dir)
                
            # Track performance metrics
            all_rcnn_times.append(logo_recog_time)
            all_siamese_times.append(logo_match_time)
            
            # Update summary if in analysis mode
            if analysis_mode:
                summary_data['successful_analyses'] += 1
                summary_data['detection_summary']['total_logos_detected'] += len(pred_boxes) if pred_boxes is not None else 0
                if pred_target:
                    summary_data['detection_summary']['total_brands_matched'] += 1
                if phish_category:
                    summary_data['detection_summary']['phishing_detected'] += 1
                else:
                    summary_data['detection_summary']['benign_results'] += 1
                    
        except KeyError as e:
            print(f"⚠️  Skipping {folder} due to KeyError: {e}")
            if analysis_mode:
                summary_data['failed_analyses'] += 1
                # Save error info
                error_file = os.path.join(sample_analysis_dir, 'error.json') if sample_analysis_dir else None
                if error_file:
                    with open(error_file, 'w') as f:
                        json.dump({'error_type': 'KeyError', 'error_message': str(e)}, f)
            continue
        except Exception as e:
            print(f"⚠️  Skipping {folder} due to error: {e}")
            if analysis_mode:
                summary_data['failed_analyses'] += 1
                # Save error info
                error_file = os.path.join(sample_analysis_dir, 'error.json') if sample_analysis_dir else None
                if error_file:
                    with open(error_file, 'w') as f:
                        json.dump({'error_type': type(e).__name__, 'error_message': str(e)}, f)
            continue

        # Save results to file
        try:
            with open(result_txt, "a+", encoding='ISO-8859-1') as f:
                result_file_write(f, folder, url, phish_category, pred_target, matched_domain, siamese_conf,
                                  logo_recog_time, logo_match_time)
        except UnicodeError:
            with open(result_txt, "a+", encoding='utf-8') as f:
                result_file_write(f, folder, url, phish_category, pred_target, matched_domain, siamese_conf,
                                  logo_recog_time, logo_match_time)
                                
        # Save prediction visualization if phishing detected
        if phish_category:
            try:
                os.makedirs(os.path.join(request_dir, folder), exist_ok=True)
                cv2.imwrite(os.path.join(request_dir, folder, "predict.png"), plotvis)
            except Exception as e:
                print(f"Warning: Could not save prediction image for {folder}: {e}")
                
        # Update total processed
        if analysis_mode:
            summary_data['total_processed'] += 1

    # Generate final summary if in analysis mode
    if analysis_mode and all_rcnn_times:
        summary_data['performance_summary'].update({
            'avg_rcnn_time': np.mean(all_rcnn_times),
            'avg_siamese_time': np.mean(all_siamese_times),
            'avg_total_time': np.mean([r + s for r, s in zip(all_rcnn_times, all_siamese_times)]),
            'min_rcnn_time': np.min(all_rcnn_times),
            'max_rcnn_time': np.max(all_rcnn_times),
            'min_siamese_time': np.min(all_siamese_times),
            'max_siamese_time': np.max(all_siamese_times),
            'std_rcnn_time': np.std(all_rcnn_times),
            'std_siamese_time': np.std(all_siamese_times)
        })
        summary_data['end_time'] = datetime.now().isoformat()
        
        # Save final summary
        with open(os.path.join(analysis_base_dir, 'analysis_summary.json'), 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        print(f"\n📈 Analysis Complete!")
        print(f"📊 Processed: {summary_data['total_processed']} samples")
        print(f"✅ Successful: {summary_data['successful_analyses']}")
        print(f"❌ Failed: {summary_data['failed_analyses']}")
        print(f"⏱️  Avg RCNN time: {summary_data['performance_summary']['avg_rcnn_time']:.3f}s")
        print(f"⏱️  Avg Siamese time: {summary_data['performance_summary']['avg_siamese_time']:.3f}s")
        print(f"🎯 Phishing detected: {summary_data['detection_summary']['phishing_detected']}")
        print(f"✅ Benign results: {summary_data['detection_summary']['benign_results']}")
        print(f"📁 Detailed analysis saved to: {analysis_base_dir}")

    print(f"\n🎉 Processing complete! Results saved to: {result_txt}")

    # Optional: export pip list
    if args.export_pip_list:
        try:
            target_dir = analysis_base_dir or os.getcwd()
            out_file = os.path.join(target_dir, 'pip_list.txt')
            with open(out_file, 'w') as f:
                proc = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
                f.write(proc.stdout)
            print(f"📦 pip list saved to {out_file}")
        except Exception as e:
            print(f"Warning: could not export pip list: {e}")