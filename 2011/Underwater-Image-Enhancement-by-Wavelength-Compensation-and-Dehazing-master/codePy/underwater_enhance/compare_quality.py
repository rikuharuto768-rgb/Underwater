
import cv2
import glob
import os
from quality_metrics import get_metrics

# Change paths to your data
input_folder = "/media/mesho/Storage/Underwater/Datasets/Algae-classification/dataset/Padina pavonica"
output_folder = "./results"

for in_file in sorted(glob.glob(f"{input_folder}/*.[jp][pn]*")):
    in_name = os.path.basename(in_file)
    out_file = os.path.join(output_folder, in_name)
    
    if not os.path.exists(out_file):
        print(f"⚠️ Missing output: {in_name}")
        continue
    
    # Load images (RGB for metrics)
    in_img = cv2.cvtColor(cv2.imread(in_file), cv2.COLOR_BGR2RGB)
    out_img = cv2.cvtColor(cv2.imread(out_file), cv2.COLOR_BGR2RGB)
    
    m_in = get_metrics(in_img)
    m_out = get_metrics(out_img)
    
    uciqe_diff = m_out['UCIQE'] - m_in['UCIQE']
    uiqm_diff = m_out['UIQM'] - m_in['UIQM']
    
    print(f"{in_name}:")
    print(f"  Input:  UCIQE={m_in['UCIQE']:.4f}, UIQM={m_in['UIQM']:.4f}")
    print(f"  Output: UCIQE={m_out['UCIQE']:.4f}, UIQM={m_out['UIQM']:.4f}")
    print(f"  Δ:      UCIQE={uciqe_diff:+.4f} (+ = improvement), UIQM={uiqm_diff:+.4f}")
    print()
