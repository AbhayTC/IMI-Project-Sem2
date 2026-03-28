├── outputs/                      # Final generated datasets
│   ├── Member1_Polyimide.csv
│   ├── Member2_PEEK.csv
│   └── Member3_PTFE.csv
├── task1_data_curation.py        # Dataset generation & feature extraction
├── task2_qspr_modeling.py        # Multi-Target MLP training
├── task3_inverse_design.py       # Differential Evolution optimization
├── task5_morgan_fingerprint.py   # ECFP4 fingerprint generation
├── task7_physics_check.py        # Physical validity & outlier checking
├── task6_config.py               # Interactive CSV export configuration (Optional) 
├── task4_output_management.py    # Aggregation & CSV export
├── task8_unified_tanimoto_score.py
├── export_config.pkl             # User configuration state
├── inv_results.pkl               # Serialized inverse design results
├── master_dataset.pkl            # Serialized main dataset (720 rows)
├── morgan_fingerprints.pkl       
├── physics_check_report.pkl      
└── qspr_model.pkl               
