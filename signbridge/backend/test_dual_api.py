import urllib.request
import json
from pathlib import Path
import numpy as np

def run_tests():
    # 1. Health
    print("=== 1. TEST /health ===")
    with urllib.request.urlopen("http://127.0.0.1:8000/health") as res:
        health = json.loads(res.read().decode("utf-8"))
        print(json.dumps(health, indent=2))

    # 2. Model Info
    print("\n=== 2. TEST /model/info ===")
    with urllib.request.urlopen("http://127.0.0.1:8000/model/info") as res:
        info = json.loads(res.read().decode("utf-8"))
        print(json.dumps(info, indent=2))

    # 3. Dynamic Sequence Predictions
    print("\n=== 3. TEST /translate/sequence (Dynamic Model) ===")
    for word in ["HELP", "PLEASE", "SORRY", "THANK_YOU"]:
        sample_file = list(Path(f"datasets/raw/member_01_dynamic/{word}").glob("*.npy"))[0]
        sample_data = np.load(sample_file).tolist()
        
        req = urllib.request.Request(
            "http://127.0.0.1:8000/translate/sequence",
            data=json.dumps({"sequence": sample_data}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode("utf-8"))
            w = data.get("word")
            t = data.get("type")
            c = data.get("confidence")
            m = data.get("message")
            print(f"Target: {word:<10} => Predicted: {w:<10} | Type: {t:<7} | Conf: {c*100:6.2f}% | Msg: {m}")

    # 4. Direct service check for static model
    print("\n=== 4. TEST Static Model via Service ===")
    from app.services.sign_recognition import SignRecognitionService
    svc = SignRecognitionService()
    for word in ["HELLO", "NO", "WATER", "YES"]:
        sample_file = list(Path(f"datasets/raw/member_01/{word}").glob("*.npy"))[0]
        arr = np.load(sample_file).astype(np.float32)
        res = svc.predict(arr)
        w = res.get("word")
        t = res.get("type")
        c = res.get("confidence")
        print(f"Target: {word:<10} => Predicted: {w:<10} | Type: {t:<7} | Conf: {c*100:6.2f}%")

if __name__ == "__main__":
    run_tests()
