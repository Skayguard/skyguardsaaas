from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

@app.post("/events/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        return {"erro": "Apenas arquivos de vídeo são permitidos."}

    contents = await file.read()
    detection = "Nenhum movimento detectado"
    frame_number = 1

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()

        while ret:
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    detection = f"Movimento detectado no frame {frame_number}"
                    cap.release()
                    return {
                        "resultado": detection,
                        "frame": frame_number,
                        "arquivo": file.filename
                    }

            frame1 = frame2
            ret, frame2 = cap.read()
            frame_number += 1

        cap.release()
        return {
            "resultado": detection,
            "frame": None,
            "arquivo": file.filename
        }

    finally:
        os.remove(tmp_path)
