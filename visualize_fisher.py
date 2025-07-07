""" 
    This file provides a transcript for visualizing fisher info. 
    Use together with Fisher Analyzer.
"""

import argparse
from models.ewc import Fisher_Analyzer

def main():
    parser = argparse.ArgumentParser(description='EWC Fisher Visualization')
    parser.add_argument('--file_path', type=str, default='logs/test/ewc_task1.pt', help='Path to the checkpoint file')
    parser.add_argument('--task_id', type=int, default=1, help='Task ID, usually set to 1 when single training')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for visualization')
    args = parser.parse_args()
    file_path = args.file_path
    task_id = args.task_id
    threshold = args.threshold

    ewc_analyzer = Fisher_Analyzer()    
    ewc_analyzer.load_fisher(file_path, task_id=task_id)
    ewc_analyzer.summarize_fisher()
    
    if threshold is not None:
        ewc_analyzer.get_high_fisher_ratio(threshold=threshold)
        ewc_analyzer.get_layerwise_high_fisher_ratio(threshold=threshold)
        ewc_analyzer.thresholded_visualize_fisher(threshold=threshold, save_dir='logs', save_fig=False)

    else:
        ewc_analyzer.visualize_fisher(save_dir='models/ewc_1/figs', save_fig=False)

if __name__ == "__main__":
    main()
