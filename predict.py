"""
High-Entropy Alloy (HEA) Phase Predictor CLI
Loads trained model from hea_phase_model.joblib and predicts crystal phase from user input.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

MODEL_FILE = 'hea_phase_model.joblib'

def load_artifacts():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        print("Please train and dump the model first (or run HEA_Phase_Prediction.ipynb).")
        sys.exit(1)
    return joblib.load(MODEL_FILE)

def format_prediction_output(pred_phase, prob_dict):
    phase_descriptions = {
        'BCC_SS': 'Body-Centered Cubic Solid Solution (Single Phase)',
        'FCC_SS': 'Face-Centered Cubic Solid Solution (Single Phase)',
        'FCC_PLUS_BCC': 'Dual Solid Solution (FCC + BCC Duplex Mixture)',
        'Im': 'Intermetallic Compound / Multiphase with Intermetallics'
    }
    
    print("\n" + "=" * 65)
    print("                    PREDICTION RESULTS")
    print("=" * 65)
    print(f"  PREDICTED PHASE:  >>> {pred_phase} <<<")
    print(f"  DESCRIPTION:      {phase_descriptions.get(pred_phase, 'Unknown')}")
    print("-" * 65)
    print("  CONFIDENCE PROBABILITIES ACROSS ALL CLASSES:")
    for phase, prob in prob_dict.items():
        bar_len = int(prob / 4)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        marker = " <== (Winner)" if phase == pred_phase else ""
        print(f"    {phase:14s} : [{bar}] {prob:5.1f}%{marker}")
    print("=" * 65 + "\n")

def run_prediction(artifacts, input_series):
    model = artifacts['model']
    le = artifacts['label_encoder']
    feature_names = artifacts['feature_names']
    
    # Ensure correct column ordering
    input_df = pd.DataFrame([input_series])[feature_names]
    
    # Predict probabilities
    probs = model.predict_proba(input_df)[0]
    pred_idx = np.argmax(probs)
    pred_phase = le.inverse_transform([pred_idx])[0]
    prob_dict = {phase: round(float(p) * 100, 2) for phase, p in zip(le.classes_, probs)}
    
    format_prediction_output(pred_phase, prob_dict)
    return pred_phase, prob_dict

def get_preset_samples():
    return {
        '1': {
            'name': 'Cantor Alloy (CoCrFeMnNi) - FCC Benchmark',
            'data': {
                'Co': 0.20, 'Cr': 0.20, 'Fe': 0.20, 'Mn': 0.20, 'Ni': 0.20,
                'Num_of_Elem': 5.0, 'Density_calc': 8.04, 'dHmix': -4.16, 'dSmix': 13.38,
                'dGmix': 450.0, 'Tm': 1630.0, 'n.Para': 5.2, 'Atom.Size.Diff': 3.3,
                'Elect.Diff': 0.11, 'VEC': 8.0, 'Route_PM': 0.0,
                'Homogenization_Temp': 1000.0, 'Annealing_Temp': 0.0
            }
        },
        '2': {
            'name': 'Refractory HEA (TiZrNbTaMo) - BCC Benchmark',
            'data': {
                'Ti': 0.20, 'Zr': 0.20, 'Nb': 0.20, 'Ta': 0.20, 'Mo': 0.20,
                'Num_of_Elem': 5.0, 'Density_calc': 9.2, 'dHmix': -7.5, 'dSmix': 13.38,
                'dGmix': 560.0, 'Tm': 2400.0, 'n.Para': 19.8, 'Atom.Size.Diff': 5.5,
                'Elect.Diff': 0.18, 'VEC': 4.6, 'Route_PM': 0.0,
                'Homogenization_Temp': 1200.0, 'Annealing_Temp': 0.0
            }
        },
        '3': {
            'name': 'Al-Rich HEA (Al1.5CoCrFeNi) - Intermetallic (B2/Im)',
            'data': {
                'Al': 0.27, 'Co': 0.18, 'Cr': 0.18, 'Fe': 0.18, 'Ni': 0.18,
                'Num_of_Elem': 5.0, 'Density_calc': 6.8, 'dHmix': -16.2, 'dSmix': 13.2,
                'dGmix': 620.0, 'Tm': 1550.0, 'n.Para': 2.5, 'Atom.Size.Diff': 6.9,
                'Elect.Diff': 0.24, 'VEC': 7.1, 'Route_PM': 0.0,
                'Homogenization_Temp': 0.0, 'Annealing_Temp': 900.0
            }
        },
        '4': {
            'name': 'Duplex HEA (Al0.5CoCrFeNi) - Dual FCC + BCC',
            'data': {
                'Al': 0.11, 'Co': 0.22, 'Cr': 0.22, 'Fe': 0.22, 'Ni': 0.22,
                'Num_of_Elem': 5.0, 'Density_calc': 7.6, 'dHmix': -8.9, 'dSmix': 13.3,
                'dGmix': 510.0, 'Tm': 1610.0, 'n.Para': 4.1, 'Atom.Size.Diff': 4.5,
                'Elect.Diff': 0.15, 'VEC': 7.8, 'Route_PM': 0.0,
                'Homogenization_Temp': 0.0, 'Annealing_Temp': 0.0
            }
        }
    }

def interactive_mode(artifacts):
    feature_names = artifacts['feature_names']
    elem_cols = artifacts['elem_cols']
    medians = artifacts['feature_medians']
    
    print("\n" + "=" * 65)
    print("      HEA PHASE PREDICTION - INTERACTIVE INPUT CONSOLE")
    print("=" * 65)
    print("Note: Press [Enter] on any field to accept the recommended default.")
    print("-" * 65)
    
    input_data = {}
    
    # 1. Elemental fractions
    print("\n[PART 1: ELEMENTAL COMPOSITION]")
    print(f"Supported elements ({len(elem_cols)}): {', '.join(elem_cols)}")
    print("Tip: Enter elements as 'Element:Fraction' separated by commas")
    print("Example: Co:0.25, Cr:0.25, Fe:0.25, Ni:0.25")
    
    comp_input = input("\nEnter alloy composition (or press Enter for Cantor alloy): ").strip()
    
    # Initialize all elements to 0.0
    for el in elem_cols:
        input_data[el] = 0.0
        
    if comp_input:
        parts = comp_input.split(',')
        total_frac = 0.0
        for p in parts:
            if ':' in p:
                el, frac = p.strip().split(':')
                el = el.strip()
                try:
                    val = float(frac.strip())
                    if el in elem_cols:
                        input_data[el] = val
                        total_frac += val
                    else:
                        print(f"Warning: Element '{el}' not recognized, ignored.")
                except ValueError:
                    pass
        print(f"Parsed total elemental fraction: {total_frac:.3f}")
    else:
        # Default Cantor composition
        print("Using default equiatomic Cantor alloy: Co=0.2, Cr=0.2, Fe=0.2, Mn=0.2, Ni=0.2")
        input_data['Co'] = 0.20
        input_data['Cr'] = 0.20
        input_data['Fe'] = 0.20
        input_data['Mn'] = 0.20
        input_data['Ni'] = 0.20

    # 2. Key Thermodynamic & Physical Properties
    print("\n[PART 2: THERMODYNAMIC & PHYSICAL DESCRIPTORS]")
    
    def prompt_float(param_name, prompt_text, default_val):
        user_val = input(f"{prompt_text} [Default: {default_val}]: ").strip()
        if not user_val:
            return float(default_val)
        try:
            return float(user_val)
        except ValueError:
            print(f"Invalid input, using default: {default_val}")
            return float(default_val)

    # Active elements count
    active_elems = sum(1 for el in elem_cols if input_data[el] > 0)
    input_data['Num_of_Elem'] = float(active_elems if active_elems > 0 else 5.0)

    input_data['VEC'] = prompt_float('VEC', 'Valence Electron Concentration (VEC)', round(medians.get('VEC', 7.5), 2))
    input_data['dHmix'] = prompt_float('dHmix', 'Enthalpy of Mixing dHmix (kJ/mol)', round(medians.get('dHmix', -5.0), 2))
    input_data['dSmix'] = prompt_float('dSmix', 'Entropy of Mixing dSmix (J/mol.K)', round(medians.get('dSmix', 13.38), 2))
    input_data['dGmix'] = prompt_float('dGmix', 'Gibbs Free Energy dGmix', round(medians.get('dGmix', 475.0), 2))
    input_data['Tm'] = prompt_float('Tm', 'Melting Temperature Tm (K)', round(medians.get('Tm', 1800.0), 1))
    input_data['Atom.Size.Diff'] = prompt_float('Atom.Size.Diff', 'Atomic Size Difference delta (%)', round(medians.get('Atom.Size.Diff', 4.0), 2))
    input_data['Elect.Diff'] = prompt_float('Elect.Diff', 'Electronegativity Difference', round(medians.get('Elect.Diff', 0.15), 3))
    input_data['Density_calc'] = prompt_float('Density_calc', 'Calculated Density (g/cm3)', round(medians.get('Density_calc', 7.8), 2))
    input_data['n.Para'] = prompt_float('n.Para', 'Thermodynamic Parameter Omega / n.Para', round(medians.get('n.Para', 9.0), 2))

    # 3. Processing conditions
    print("\n[PART 3: PROCESSING & HEAT TREATMENT]")
    route_choice = input("Synthesis Route: 1 for Arc-Melting (AC) / 2 for Powder Met (PM) [Default: 1 (AC)]: ").strip()
    input_data['Route_PM'] = 1.0 if route_choice == '2' else 0.0

    input_data['Homogenization_Temp'] = prompt_float('Homogenization_Temp', 'Homogenization Temperature in C (0 if as-cast)', 0.0)
    input_data['Annealing_Temp'] = prompt_float('Annealing_Temp', 'Annealing Temperature in C (0 if as-cast)', 0.0)

    # Ensure all required columns are filled
    full_sample = {}
    for col in feature_names:
        full_sample[col] = input_data.get(col, medians.get(col, 0.0))
        
    run_prediction(artifacts, full_sample)

def main():
    artifacts = load_artifacts()
    
    print("\n" + "=" * 65)
    print("   HIGH-ENTROPY ALLOY (HEA) PHASE PREDICTION SYSTEM")
    print(f"   Loaded Model: {type(artifacts['model']).__name__} (Features: {len(artifacts['feature_names'])})")
    print("=" * 65)
    
    while True:
        print("\nSelect an option:")
        print("  1. Interactive Manual Input (Enter composition & properties)")
        print("  2. Quick Benchmark Presets (Cantor, Refractory, Intermetallic, Duplex)")
        print("  3. Predict from a CSV file")
        print("  4. Exit")
        
        choice = input("\nEnter choice [1-4]: ").strip()
        
        if choice == '1':
            interactive_mode(artifacts)
        elif choice == '2':
            presets = get_preset_samples()
            print("\nAvailable Presets:")
            for k, p in presets.items():
                print(f"  {k}. {p['name']}")
            preset_choice = input("\nSelect preset [1-4]: ").strip()
            if preset_choice in presets:
                p_info = presets[preset_choice]
                print(f"\nEvaluating: {p_info['name']}...")
                # Fill missing with medians
                full_sample = artifacts['feature_medians'].copy()
                # Zero out elements first
                for el in artifacts['elem_cols']:
                    full_sample[el] = 0.0
                full_sample.update(p_info['data'])
                run_prediction(artifacts, full_sample)
            else:
                print("Invalid preset choice.")
        elif choice == '3':
            csv_path = input("Enter path to CSV file: ").strip()
            if os.path.exists(csv_path):
                try:
                    user_df = pd.read_csv(csv_path)
                    print(f"Loaded {len(user_df)} rows. Predicting...")
                    feature_names = artifacts['feature_names']
                    # Fill missing columns with medians
                    for col in feature_names:
                        if col not in user_df.columns:
                            user_df[col] = artifacts['feature_medians'].get(col, 0.0)
                    predictions = []
                    for _, row in user_df.iterrows():
                        pred_p, _ = run_prediction(artifacts, row[feature_names])
                        predictions.append(pred_p)
                    user_df['Predicted_Phase'] = predictions
                    out_path = 'predictions_output.csv'
                    user_df.to_csv(out_path, index=False)
                    print(f"Results saved to '{out_path}'")
                except Exception as e:
                    print(f"Error processing CSV: {e}")
            else:
                print("File not found.")
        elif choice == '4':
            print("Exiting HEA Phase Predictor. Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, or 4.")

if __name__ == '__main__':
    main()
