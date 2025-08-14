# Phishpedia Enhanced Analysis Suite

This enhanced version of Phishpedia provides comprehensive analysis features including detailed performance monitoring, system resource tracking, visual outputs, and per-folder analysis with overall statistics.

## 🆕 New Features

### 1. **Analysis Mode** (`--analysis_mode`)
- **Real-time System Monitoring**: CPU, memory, and GPU usage tracking at each processing stage
- **Individual Processing Metrics**: Detailed timing for RCNN and Siamese models per image
- **Visual Output Generation**: Annotated images showing RCNN detections and Siamese matches
- **Comprehensive JSON Analysis**: Detailed metrics saved for each processed image
- **Error Logging**: Robust error capture with detailed traces

### 2. **Enhanced Visual Outputs**
- **RCNN Detections**: Green bounding boxes showing all detected logo regions
- **Siamese Matches**: Red bounding boxes highlighting brand matches with confidence scores
- **Detailed Labels**: Clear labeling of model outputs and confidence scores
- **Legend and Metadata**: Image annotations with processing information

### 3. **Enhanced Analysis Reporter** (`enhanced_analysis_reporter.py`)
- **Per-Folder Analysis**: Detailed breakdown of each processed folder
- **Overall Statistics**: Comprehensive averages and performance metrics
- **CSV Export**: Detailed spreadsheet-compatible data export
- **Performance Visualizations**: Automated charts and graphs
- **Resource Usage Analysis**: CPU and memory consumption patterns

## 🚀 Quick Start

### Step 1: Run Analysis Mode
```bash
# Activate environment
.\phishpedia_env\Scripts\activate

# Basic analysis on test sites
python phishpedia.py --folder ./datasets/test_sites --analysis_mode

# Limited sample analysis (for testing)
python phishpedia.py --folder ./datasets/test_sites --analysis_mode --max_samples 5
```

### Step 2: Generate Comprehensive Reports
```bash
# Generate detailed analysis and visualizations
python enhanced_analysis_reporter.py analysis_20250805 --output-dir detailed_results
```

## 📊 Output Structure

### Analysis Mode Output (`analysis_YYYYMMDD/`)
```
analysis_20250805/
├── analysis_summary.json          # Overall processing summary
├── [folder_1]/                    # Individual analysis folders
│   ├── analysis.json              # Comprehensive analysis data
│   └── detection_visual.png       # Annotated image with detections
├── [folder_2]/
│   ├── analysis.json
│   └── detection_visual.png
└── ...
```

### Enhanced Analysis Output (`detailed_results/`)
```
detailed_results/
├── detailed_folder_analysis.csv   # Spreadsheet with all metrics
└── visualizations/
    └── detailed_performance_analysis.png  # Performance charts
```

## 📈 Per-Folder Analysis Details

For each processed folder, the system captures:

### 🖼️ **Image Information**
- Resolution and file size
- Format and metadata

### 💻 **System Resource Usage**
- CPU utilization at start, after RCNN, and final stages
- RAM usage in GB and percentage
- GPU memory allocation (if available)

### 🤖 **Model Performance**
- **RCNN Model**: Processing time, success status, number of logos detected
- **Siamese Model**: Processing time, matched brand, confidence score
- **Total Processing Time**: End-to-end analysis duration

### 🎯 **Detection Results**
- Logo bounding box coordinates and dimensions
- Final classification (phishing/benign)
- Brand matches and confidence scores
- Detailed logo positioning data

### ⚠️ **Error Tracking**
- Exception logs and error details
- Success/failure status for each model

## 📊 Overall Statistics & Averages

The enhanced reporter generates comprehensive statistics including:

### 📈 **Processing Summary**
- Total folders processed
- RCNN and Siamese success rates
- Phishing vs. benign detection rates

### ⏱️ **Performance Metrics**
- Average, median, min/max processing times
- Images processed per second
- Standard deviation and performance consistency

### 💻 **Resource Usage Analysis**
- Average CPU and RAM consumption
- Peak resource usage
- Resource efficiency patterns

### 🔍 **Detection Analysis**
- Average logos detected per image
- Brand detection frequency
- Confidence score distributions
- Most common detected brands

## 🖼️ Visual Output Features

### Enhanced Detection Visualization
Each processed image generates an annotated visualization showing:

1. **Title Bar**: Phishpedia Analysis header with model information
2. **RCNN Results**: Green bounding boxes around all detected logo regions
   - Labeled with detection confidence (if available)
   - Numbered for easy reference
3. **Siamese Results**: Red bounding boxes around matched brands
   - Brand name and confidence score
   - Clear distinction from RCNN detections
4. **Legend**: Color-coded explanation of visualization elements
5. **Model Status**: Summary of RCNN and Siamese results

## � Sample Analysis Output

### Individual Folder Report
```
🔍 FOLDER 1: example-phishing-site
----------------------------------------
📅 Processed: 2025-08-05T12:20:59.915608
🖼️  Image: 1366x918 | 86,392 bytes

💻 SYSTEM RESOURCE USAGE:
   START: CPU 24.5% | RAM 95.9% (7.41GB)
   AFTER_RCNN: CPU 17.6% | RAM 91.8% (7.10GB)
   FINAL: CPU 21.6% | RAM 88.3% (6.82GB)

🤖 MODEL PERFORMANCE:
   RCNN: ✅ SUCCESS | 3.661s | 2 logos detected
   SIAMESE: ✅ SUCCESS | 0.512s | Brand: Microsoft (0.748)

🎯 DETECTION RESULTS:
   Final Result: 🚨 PHISHING (brand_match)
   Total Processing Time: 4.173s
   Logo Coordinates:
     Logo 1: (12,13) 93x45px
     Logo 2: (12,15) 40x40px
```

### Overall Statistics
```
📊 OVERALL STATISTICS & AVERAGES
==================================================
📈 PROCESSING SUMMARY:
   Total Folders Processed: 15
   RCNN Success Rate: 15/15 (100.0%)
   Siamese Success Rate: 14/15 (93.3%)
   Phishing Detected: 8 (53.3%)
   Benign Results: 7 (46.7%)

⏱️  PERFORMANCE METRICS:
   RCNN Processing Time:
     Average: 3.323s | Median: 3.199s
     Min/Max: 2.108s / 5.661s
     Images per second: 0.30

💻 SYSTEM RESOURCE USAGE:
   CPU Usage: Average 22.5% | Peak 45.2%
   RAM Usage: Average 85.4% | Peak 96.1%

🔍 DETECTION ANALYSIS:
   Logos per Image: Average 3.67
   Confidence Scores: Average 0.769
```

## 📊 CSV Export Format

The detailed CSV export includes columns for:
- `folder`: Folder name
- `timestamp`: Processing timestamp
- `image_resolution`: Image dimensions
- `image_size_bytes`: File size
- `rcnn_time`: RCNN processing time
- `rcnn_success`: RCNN success status
- `rcnn_detections`: Number of logos detected
- `siamese_time`: Siamese processing time
- `siamese_success`: Siamese success status
- `siamese_brand`: Matched brand (if any)
- `siamese_confidence`: Confidence score
- `total_time`: Total processing time
- `final_result`: Classification result
- `is_phishing`: Boolean phishing flag
- `cpu_start`, `cpu_final`: CPU usage
- `ram_start`, `ram_final`: RAM usage
- `error_count`: Number of errors

## 🎯 Use Cases

### 1. **Performance Analysis**
- Monitor processing speed and resource usage
- Identify bottlenecks in RCNN vs Siamese processing
- Optimize system configuration for better performance

### 2. **Model Evaluation**
- Compare success rates across different image types
- Analyze confidence score patterns
- Identify problematic scenarios for model improvement

### 3. **Quality Assurance**
- Visual verification of detection accuracy
- Error pattern analysis
- Success rate monitoring

### 4. **Research and Development**
- Detailed performance baselines
- Comparative analysis across datasets
- Model behavior analysis

## 🔧 Command Line Options

```bash
# Main Analysis
python phishpedia.py --folder <input_folder> [options]

Required:
  --folder                Input folder containing test sites

Analysis Options:
  --analysis_mode         Enable comprehensive analysis mode
  --max_samples N         Limit processing to N samples
  --output_txt FILE       Custom results text file name

# Enhanced Reporting
python enhanced_analysis_reporter.py <analysis_dir> [options]

Required:
  analysis_dir           Directory containing analysis results

Reporting Options:
  --output-dir DIR       Output directory for reports
```

## 🚦 System Requirements

- **Python Environment**: phishpedia_env with PyTorch
- **Required Packages**: matplotlib, pandas, seaborn (auto-installed)
- **Storage**: ~10MB per analyzed image for full analysis
- **Memory**: 8GB+ RAM recommended for analysis mode

## � Example Workflow

1. **Activate Environment**:
   ```bash
   .\phishpedia_env\Scripts\activate
   ```

2. **Run Analysis**:
   ```bash
   python phishpedia.py --folder ./datasets/test_sites --analysis_mode --max_samples 10
   ```

3. **Generate Reports**:
   ```bash
   python enhanced_analysis_reporter.py analysis_20250805 --output-dir final_results
   ```

4. **Review Results**:
   - Check individual folder visualizations
   - Review CSV data for detailed metrics
   - Examine performance charts

This enhanced analysis suite transforms Phishpedia from a basic detection tool into a comprehensive analysis platform suitable for research, evaluation, and production monitoring.
