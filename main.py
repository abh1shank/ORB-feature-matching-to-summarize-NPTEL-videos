from ultralytics import YOLO
from pytube import YouTube
import cv2
import os
import numpy as np
import io
import img2pdf
from PIL import Image

model=YOLO('yolov8n.pt')
def get_video(link):
    pass
    yt = YouTube(link)
    stream = yt.streams.get_highest_resolution()
    download_dir = ''
    stream.download(download_dir)
    print("downloaded")


def get_frames_array():
    vid_path = 'stack_test.mp4'
    print(os.path.exists(vid_path))
    cap = cv2.VideoCapture(vid_path)
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    frames=np.array(frames)
    return frames

def check_duplicates(frames):
    similarity_threshold=0.8
    unique_frames=[]
    frames_to_ignore=[]
    for i in range(len(frames)):
        if teacher_present(frames[i]):
            continue
        if i in frames_to_ignore:
            continue
        for j in range(len(frames)):
            similarity_score = ORB_point_checking(frames[i], frames[j])
            if (i!=j and similarity_score > similarity_threshold):
                frames_to_ignore.append(j)
        if i not in frames_to_ignore:
            unique_frames.append(frames[i])

    return unique_frames

def teacher_present(frame)->bool:
        print("checking for teacher")
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=model.predict([frame])
        for result in results:
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            truth_array = boxes.cls == 0.
            truth_array = np.array(truth_array)
            if(len(truth_array)==0):
                return 0
            if truth_array[0]:
                return 1
            else: return 0
        return 0

def ORB_point_checking(img1,img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches)==0:
        return 0
    similarity_score = len(similar_regions) / len(matches)
    return similarity_score

def check_similarity(frames):
    similarity_threshold=0.9
    capped_frames=[]
    for i in range(len(frames)-1):
        similarity_score=ORB_point_checking(frames[i],frames[i+1])
        if(similarity_score<similarity_threshold):
            capped_frames.append(frames[i])
    return capped_frames

def to_pdf(frames, output_pdf):
    white_screen = np.ones(shape=(360, 640))
    target_height, target_width = 720, 1280
    jpeg_files = []

    for i, frame in enumerate(frames):
        padded_frame = np.pad(frame, ((0, target_height - frame.shape[0]), (0, target_width - frame.shape[1]), (0, 0)), mode='constant', constant_values=1)
        pil_image = Image.fromarray((padded_frame * 255).astype(np.uint8))
        jpeg_file = f"frame_{i}.jpg"
        pil_image.save(jpeg_file, format='JPEG')
        jpeg_files.append(jpeg_file)

    with open(output_pdf, "wb") as pdf_file:
        pdf_file.write(img2pdf.convert(jpeg_files))

    # Clean up JPEG files
    for jpeg_file in jpeg_files:
        os.remove(jpeg_file)

def frames_to_images(frames, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_folder, f"frame_{i}.jpg")
        cv2.imwrite(output_path, frame)

    print("Frames converted to images successfully.")


def main():
    full_frames=get_frames_array()
    print(f"total frames are {len(full_frames)}")
    capped_frames=check_similarity(full_frames)
    print(f"Number of frames after difference cutoff {len(capped_frames)}")
    capped_frames=check_duplicates(capped_frames)
    print(f"Number of frames after duplicate checking {len(capped_frames)}")

    op_loc='C:/Users/saran/Desktop/MLDL/NPTEL Summarizer'
    to_pdf(capped_frames,op_loc+'/op.pdf')
    frames_to_images(capped_frames,op_loc)
    print("PDF saved successfully")

def debug():
    print(os.path.exists('C:/Users/saran/Desktop/MLDL/NPTEL Summarizer'))

if __name__ == '__main__':
    debug()
    main()

