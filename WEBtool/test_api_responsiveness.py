"""
Quick script to test FastAPI responsiveness using analyze-bytes endpoint
Tests shot.png files from each folder in datasets/test_sites
Usage:
  - Start the server:
      uvicorn WEBtool.api_fast:app --host 0.0.0.0 --port 8000 --reload
  - Run this script:
      python WEBtool/test_api_responsiveness.py --server http://localhost:8000 --root datasets/test_sites

Collects comprehensive statistics: avg/min/max latency, success rates, and detailed results.
"""
import os
import time
import argparse
import json
import requests
import statistics
from urllib.parse import quote


def read_url_from_info(info_path: str) -> str:
    """Read URL from info.txt file"""
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Handle different formats (JSON or plain text)
            if content.startswith('{'):
                try:
                    import json
                    data = json.loads(content)
                    return data.get('url', '')
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract URL manually
                    import re
                    url_match = re.search(r"['\"]url['\"]:\s*['\"]([^'\"]+)['\"]", content)
                    if url_match:
                        return url_match.group(1)
                    return ''
            else:
                return content
    except Exception as e:
        print(f"Warning: Could not read URL from {info_path}: {e}")
        return ''


def iter_test_sites(root: str):
    """Iterate through test site folders, looking for shot.png + info.txt pairs"""
    for folder_name in os.listdir(root):
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        shot_path = os.path.join(folder_path, 'shot.png')
        info_path = os.path.join(folder_path, 'info.txt')
        
        if os.path.exists(shot_path) and os.path.exists(info_path):
            url = read_url_from_info(info_path)
            yield {
                'folder': folder_name,
                'shot_path': shot_path,
                'info_path': info_path,
                'url': url
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://localhost:8000')
    ap.add_argument('--root', default='datasets/test_sites')
    ap.add_argument('--preprocess', default='true')
    ap.add_argument('--siamese-threshold', default='0.78', help='Siamese matching threshold')
    ap.add_argument('--timeout', type=int, default=120, help='Request timeout in seconds')
    args = ap.parse_args()

    # Use analyze-bytes endpoint
    url_api = args.server.rstrip('/') + '/api/analyze-bytes'
    results = []
    successful_results = []
    failed_results = []
    
    print(f"Testing FastAPI analyze-bytes endpoint: {url_api}")
    print(f"Root directory: {args.root}")
    print(f"Preprocess: {args.preprocess}")
    print(f"Siamese threshold: {args.siamese_threshold}")
    print("=" * 60)
    
    t0 = time.time()
    n = 0
    
    for site_info in iter_test_sites(args.root):
        n += 1
        folder = site_info['folder']
        shot_path = site_info['shot_path']
        url = site_info['url']
        
        print(f"[{n:3d}] Testing {folder}...")
        
        # Prepare query parameters
        params = {
            'url': url,
            'preprocess': args.preprocess.lower(),
            'siamese_threshold': args.siamese_threshold,
            'align': 'true',
            'archeck': 'false'
        }
        
        # Read image as bytes
        try:
            with open(shot_path, 'rb') as f:
                img_bytes = f.read()
        except Exception as e:
            result_data = {
                'folder': folder,
                'shot_path': shot_path,
                'url': url,
                'latency_sec': 0.0,
                'success': False,
                'error': f'Could not read image: {str(e)}',
                'predicted_brand': None,
                'confidence': None,
                'bboxes': None
            }
            results.append(result_data)
            failed_results.append(result_data)
            print(f"      ❌ Error reading image: {e}")
            continue
        
        # Send request
        t1 = time.time()
        try:
            response = requests.post(
                url_api, 
                params=params, 
                data=img_bytes,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=args.timeout
            )
            response.raise_for_status()
            out = response.json()
            dt = time.time() - t1
            
            # Extract key results from simplified response
            success = out.get('success', False)
            predicted_brand = out.get('predicted_brand', None)
            confidence = out.get('confidence', None)
            bboxes = out.get('bboxes', [])
            detection_time = out.get('detection_time', None)
            
            result_data = {
                'folder': folder,
                'shot_path': shot_path,
                'url': url,
                'latency_sec': dt,
                'success': success,
                'predicted_brand': predicted_brand,
                'confidence': confidence,
                'bboxes': bboxes,
                'detection_time': detection_time,
                'full_response': out
            }
            
            if success:
                successful_results.append(result_data)
                conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
                print(f"      ✅ {dt:.2f}s | Brand: {predicted_brand} | Conf: {conf_str} | Boxes: {len(bboxes)}")
            else:
                failed_results.append(result_data)
                error_msg = out.get('error', 'Unknown error')
                print(f"      ❌ {dt:.2f}s | Failed: {error_msg}")
                
        except requests.exceptions.Timeout:
            dt = args.timeout
            result_data = {
                'folder': folder,
                'shot_path': shot_path,
                'url': url,
                'latency_sec': dt,
                'success': False,
                'error': 'Request timeout',
                'predicted_brand': None,
                'confidence': None,
                'bboxes': None
            }
            failed_results.append(result_data)
            print(f"      ⏰ Timeout after {dt}s")
            
        except Exception as e:
            dt = time.time() - t1
            result_data = {
                'folder': folder,
                'shot_path': shot_path,
                'url': url,
                'latency_sec': dt,
                'success': False,
                'error': str(e),
                'predicted_brand': None,
                'confidence': None,
                'bboxes': None
            }
            failed_results.append(result_data)
            print(f"      ❌ {dt:.2f}s | Error: {str(e)}")
        
        results.append(result_data)

    total_time = time.time() - t0
    
    # Calculate comprehensive statistics
    if results:
        latencies = [r['latency_sec'] for r in results]
        successful_latencies = [r['latency_sec'] for r in successful_results]
        
        stats = {
            'test_info': {
                'endpoint': url_api,
                'root_directory': args.root,
                'preprocess': args.preprocess,
                'siamese_threshold': float(args.siamese_threshold),
                'timeout': args.timeout,
                'total_test_time_sec': total_time
            },
            'counts': {
                'total_tests': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate_percent': (len(successful_results) / len(results)) * 100
            },
            'latency_stats': {
                'all_requests': {
                    'avg_sec': statistics.mean(latencies),
                    'min_sec': min(latencies),
                    'max_sec': max(latencies),
                    'median_sec': statistics.median(latencies),
                    'std_dev_sec': statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                }
            },
            'logo_detection': {
                'with_logo_detected': len([r for r in successful_results if r.get('predicted_brand')]),
                'without_logo_detected': len([r for r in successful_results if not r.get('predicted_brand')]),
                'avg_confidence': statistics.mean([r['confidence'] for r in successful_results if r.get('confidence')]) if any(r.get('confidence') for r in successful_results) else None
            }
        }
        
        if successful_latencies:
            stats['latency_stats']['successful_only'] = {
                'avg_sec': statistics.mean(successful_latencies),
                'min_sec': min(successful_latencies),
                'max_sec': max(successful_latencies),
                'median_sec': statistics.median(successful_latencies),
                'std_dev_sec': statistics.stdev(successful_latencies) if len(successful_latencies) > 1 else 0.0
            }
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"🔢 Test Counts:")
        print(f"   Total tests: {stats['counts']['total_tests']}")
        print(f"   Successful: {stats['counts']['successful']} ({stats['counts']['success_rate_percent']:.1f}%)")
        print(f"   Failed: {stats['counts']['failed']}")
        
        print(f"\n⏱️  Response Time Statistics (All Requests):")
        all_stats = stats['latency_stats']['all_requests']
        print(f"   Average: {all_stats['avg_sec']:.3f}s")
        print(f"   Minimum: {all_stats['min_sec']:.3f}s")
        print(f"   Maximum: {all_stats['max_sec']:.3f}s")
        print(f"   Median:  {all_stats['median_sec']:.3f}s")
        print(f"   Std Dev: {all_stats['std_dev_sec']:.3f}s")
        
        if 'successful_only' in stats['latency_stats']:
            success_stats = stats['latency_stats']['successful_only']
            print(f"\n✅ Response Time Statistics (Successful Only):")
            print(f"   Average: {success_stats['avg_sec']:.3f}s")
            print(f"   Minimum: {success_stats['min_sec']:.3f}s")
            print(f"   Maximum: {success_stats['max_sec']:.3f}s")
            print(f"   Median:  {success_stats['median_sec']:.3f}s")
            print(f"   Std Dev: {success_stats['std_dev_sec']:.3f}s")
        
        logo_stats = stats['logo_detection']
        print(f"\n🎯 Logo Detection Results:")
        print(f"   With logo detected: {logo_stats['with_logo_detected']}")
        print(f"   Without logo detected: {logo_stats['without_logo_detected']}")
        if logo_stats['avg_confidence']:
            print(f"   Average confidence: {logo_stats['avg_confidence']:.3f}")
        
        print(f"\n🕒 Total test execution time: {total_time:.2f}s")
        
        # Top performers and failures
        if successful_results:
            fastest = min(successful_results, key=lambda x: x['latency_sec'])
            slowest = max(successful_results, key=lambda x: x['latency_sec'])
            print(f"\n🏆 Performance Highlights:")
            print(f"   Fastest successful: {fastest['folder']} ({fastest['latency_sec']:.3f}s)")
            print(f"   Slowest successful: {slowest['folder']} ({slowest['latency_sec']:.3f}s)")
        
        if failed_results:
            print(f"\n❌ Failed Tests:")
            for fail in failed_results[:5]:  # Show first 5 failures
                print(f"   {fail['folder']}: {fail.get('error', 'Unknown error')}")
            if len(failed_results) > 5:
                print(f"   ... and {len(failed_results) - 5} more failures")
    
    else:
        stats = {'error': 'No test results collected'}
        print("❌ No test results collected!")

    # Save detailed results
    output_data = {
        'summary': stats,
        'detailed_results': results,
        'successful_results': successful_results,
        'failed_results': failed_results
    }
    
    output_file = 'api_responsiveness_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
