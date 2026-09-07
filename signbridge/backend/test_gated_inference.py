import numpy as np
import tensorflow as tf
from pathlib import Path

stat_m = tf.keras.models.load_model('app/models/static_model_v2.keras')
dyn_m = tf.keras.models.load_model('app/models/dynamic_model_v2.keras')
s_classes = ['HELLO', 'NO', 'WATER', 'YES']
d_classes = ['HELP', 'PLEASE', 'SORRY', 'THANK_YOU']

def predict_gated(buffer_30, current_frame, threshold=0.70, motion_thresh=0.07):
    # Check motion energy across the sequence: max Euclidean distance from initial frame
    max_disp = float(np.max(np.linalg.norm(buffer_30 - buffer_30[0], axis=1)))
    
    # If the hand has significant motion, evaluate as a dynamic gesture
    if max_disp >= motion_thresh:
        d_p = dyn_m.predict(np.expand_dims(buffer_30, 0), verbose=0)[0]
        d_idx = int(np.argmax(d_p))
        d_conf = float(d_p[d_idx])
        d_word = d_classes[d_idx]
        
        if d_conf >= threshold:
            return {'word': d_word, 'type': 'dynamic', 'conf': d_conf, 'motion': max_disp}
        else:
            return {'word': 'uncertain', 'type': 'dynamic', 'conf': d_conf, 'motion': max_disp}
    
    # If the hand is stationary, evaluate as a static sign
    s_p = stat_m.predict(np.expand_dims(current_frame, 0), verbose=0)[0]
    s_idx = int(np.argmax(s_p))
    s_conf = float(s_p[s_idx])
    s_word = s_classes[s_idx]
    
    if s_conf >= threshold:
        return {'word': s_word, 'type': 'static', 'conf': s_conf, 'motion': max_disp}
    
    return {'word': 'uncertain', 'type': 'static', 'conf': s_conf, 'motion': max_disp}

def run():
    print("=== TESTING STATIONARY HANDS (Previous Failure Mode: Repeated THANK_YOU) ===")
    for w in ['HELLO', 'YES', 'NO', 'WATER']:
        f = list(Path(f'datasets/raw/member_01/{w}').glob('*.npy'))[0]
        frame = np.load(f).astype(np.float32)
        # 30 stationary frames with camera jitter
        buf = np.tile(frame, (30, 1)) + np.random.normal(0, 0.001, (30, 63)).astype(np.float32)
        res = predict_gated(buf, frame)
        word = res['word']
        ptype = res['type']
        conf = res['conf']
        mot = res['motion']
        print(f"Target: {w:<10} => Predicted: {word:<10} | Type: {ptype:<7} | Conf: {conf*100:6.2f}% | Motion: {mot:.4f}")

    print("\n=== TESTING REAL DYNAMIC GESTURES ===")
    for w in ['THANK_YOU', 'PLEASE', 'SORRY', 'HELP']:
        f = list(Path(f'datasets/raw/member_01_dynamic/{w}').glob('*.npy'))[0]
        seq = np.load(f).astype(np.float32)
        res = predict_gated(seq, seq[-1])
        word = res['word']
        ptype = res['type']
        conf = res['conf']
        mot = res['motion']
        print(f"Target: {w:<10} => Predicted: {word:<10} | Type: {ptype:<7} | Conf: {conf*100:6.2f}% | Motion: {mot:.4f}")

if __name__ == "__main__":
    run()
