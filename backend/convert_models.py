import pickle
import xgboost as xgb
import os


def convert_model(input_path, output_path):
    """Convert old XGBoost model to new format"""
    try:
        print(f"Loading model from: {input_path}")

        # Load the old model
        with open(input_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # Extract model
        if isinstance(loaded_data, dict):
            model = loaded_data.get('model', loaded_data)
            feature_names = loaded_data.get('feature_names')
        else:
            model = loaded_data
            feature_names = None

        # Get booster and save in new format
        booster = model.get_booster()

        # Save booster in JSON format (more compatible)
        json_path = output_path.replace('.pkl', '.json')
        booster.save_model(json_path)
        print(f"✅ Saved JSON model: {json_path}")

        # Also create a new pickle with CPU settings
        new_model = xgb.XGBRegressor()
        new_model._Booster = booster
        new_model.set_params(
            tree_method='hist',
            predictor='cpu_predictor'
        )

        # Save with feature names if available
        if feature_names:
            save_data = {
                'model': new_model,
                'feature_names': feature_names
            }
        else:
            save_data = new_model

        with open(output_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"✅ Converted model saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


# Convert both models
print("="*60)
print("Converting XGBoost Models to Compatible Format")
print("="*60)

models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Convert day model
day_success = convert_model(
    os.path.join(models_dir, 'xgb_day_model.pkl'),
    os.path.join(models_dir, 'xgb_day_model_new.pkl')
)

# Convert hour model
hour_success = convert_model(
    os.path.join(models_dir, 'xgb_hour_model.pkl'),
    os.path.join(models_dir, 'xgb_hour_model_new.pkl')
)

if day_success and hour_success:
    print("\n✅ All models converted successfully!")
    print("\nNext steps:")
    print("1. Backup old models:")
    print("   mv models/xgb_day_model.pkl models/xgb_day_model_old.pkl")
    print("   mv models/xgb_hour_model.pkl models/xgb_hour_model_old.pkl")
    print("2. Rename new models:")
    print("   mv models/xgb_day_model_new.pkl models/xgb_day_model.pkl")
    print("   mv models/xgb_hour_model_new.pkl models/xgb_hour_model.pkl")
else:
    print("\n❌ Some models failed to convert. Check errors above.")
