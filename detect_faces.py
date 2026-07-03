import sys
import cv2
import numpy as np
from retinaface import RetinaFace


def detect_faces(image_path: str) -> list[dict]:
    faces = RetinaFace.detect_faces(image_path)
    if not isinstance(faces, dict):
        return []
    results = []
    for face_id, face_data in faces.items():
        results.append({
            "id": face_id,
            "confidence": face_data["score"],
            "box": face_data["facial_area"],
            "landmarks": face_data["landmarks"],
        })
    return results


def draw_detections(image: np.ndarray, faces: list[dict], padding: float = 0.35) -> np.ndarray:
    output = image.copy()
    h, w = output.shape[:2]
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        confidence = face["confidence"]
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * padding), int(bh * padding)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        for name, (lx, ly) in face["landmarks"].items():
            cv2.circle(output, (int(lx), int(ly)), 3, (0, 0, 255), -1)
    return output


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_faces.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.jpg"

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image '{image_path}'")
        sys.exit(1)

    print(f"Detecting faces in '{image_path}'...")
    faces = detect_faces(image_path)
    print(f"Found {len(faces)} face(s)")

    for face in faces:
        x1, y1, x2, y2 = face["box"]
        print(
            f"  {face['id']}: confidence={face['confidence']:.4f} "
            f"box=({x1},{y1})-({x2},{y2})"
        )

    output = draw_detections(image, faces)
    cv2.imwrite(output_path, output)
    print(f"Saved annotated image to '{output_path}'")


if __name__ == "__main__":
    main()
