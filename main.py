import cv2
import numpy as np
import requests
import json
import warnings
import threading

from googletrans import Translator
from PIL import Image, ImageFont, ImageDraw
from openvino.runtime import Core
from gtts import gTTS
from playsound import playsound

warnings.filterwarnings("ignore")

url     = "https://openapi.naver.com/v1/papago/n2mt"
headers = {
    "Content-Type"         : "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Naver-Client-Id"    : "your_client_id",
    "X-Naver-Client-Secret": "your_client_key"
}

font            = ImageFont.truetype("NanumGothic.ttf",20)
ie              = Core()
result_img      = None
refined_text    = ""
translated_text = ""
translator      = Translator()
model_path      = "models/"

detection_model            = ie.read_model(model=model_path+"horizontal-text-detection-0001.xml", weights=model_path+"horizontal-text-detection-0001.bin")
detection_compiled_model   = ie.compile_model(model=detection_model, device_name="CPU")
detection_input_layer      = detection_compiled_model.input(0)
output_key                 = detection_compiled_model.output("boxes")

recognition_model          = ie.read_model(model=model_path+"text-recognition-resnet-fc.xml", weights=model_path+"text-recognition-resnet-fc.bin")
recognition_compiled_model = ie.compile_model(model=recognition_model, device_name="CPU")
recognition_input_layer    = recognition_compiled_model.input(0)
recognition_output_layer   = recognition_compiled_model.output(0)

letters = "~0123456789abcdefghijklmnopqrstuvwxyz"
camera  = cv2.VideoCapture(0)

def multiply_by_ratio(ratio_x, ratio_y, box):
        return [
            max(shape * ratio_y, 10) if idx % 2 else shape * ratio_x
            for idx, shape in enumerate(box[:-1])]

def run_preprocesing_on_crop(crop, net_shape):
    temp_img = cv2.resize(crop, net_shape)
    temp_img = temp_img.reshape((1,) * 2 + temp_img.shape)
    return temp_img

def convert_result_to_image(rgb_image, resized_image,boxes, threshold=0.3, conf_labels=True):
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "white": (255, 255, 255), "black": (0, 0, 0)}
    (real_y, real_x), (resized_y, resized_x) = rgb_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    img_pil = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(img_pil)
    for box, annotation in boxes:
            conf = box[-1]
            if conf > threshold:
                (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, box))
                draw.rectangle([(x_min, y_min),(x_max, y_max)],outline=colors["green"], width=3)
                if conf_labels:
                    (text_w, text_h) = draw.textsize(annotation,font=font)
                    draw.rectangle([(x_min, y_min - text_h - 10), (x_min + text_w, y_min - 10)],fill=colors["white"],) 
                    #draw.rectangle([(x_min, y_min), (x_max, y_max)],fill=colors["white"],)              
                    draw.text((x_min,y_min-35),f"{annotation}",fill=colors["black"],font=font)
                    #draw.text((x_min+10,y_min+10),f"{annotation}",fill=colors["black"],font=font)               
    rgb_image = np.array(img_pil)
    return rgb_image

def translation(input):
    global translated_text
    def translation_thread_function():
        target_text     = input
        data            = {"source": "en", "target": "ko", "text": target_text}
        response        = requests.post(url,headers=headers, data=data)
        response_data   = json.loads(response.text)
        translated_text = response_data["message"]["result"]["translatedText"]

        print(target_text)
        print(translated_text)
        tts = gTTS(text=f"{translated_text}",lang="ko")
        tts.save("result.mp3")
    translation_thread = threading.Thread(target=translation_thread_function)
    translation_thread.start()
    translation_thread.join()          

def play():
    def play_thread_function():       
        playsound("result.mp3")
    play_thread = threading.Thread(target=play_thread_function)       
    play_thread.start()

def real_time_translation(input):
    global translated_annotations
    def rtt_thread_function():              
        text            = "".join(input)
        target_text     = text
        data            = {"source": "en", "target": "ko", "text": target_text}
        response        = requests.post(url,headers=headers, data=data)
        response_data   = json.loads(response.text)
        translated_text = response_data["message"]["result"]["translatedText"]
        #translated_text = translator.translate(text,dest='ko')
        #translated_text = translated_text.text
        translated_annotations.append(translated_text)
    rtt_thread = threading.Thread(target=rtt_thread_function)
    rtt_thread.start()
    
def processing_thread_function(image):
    global refined_text, frame, translated_annotations
    N, C, H, W = detection_input_layer.shape
    resized_image = cv2.resize(image, (W, H))
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

    boxes = detection_compiled_model([input_image])[output_key]
    boxes = boxes[~np.all(boxes == 0, axis=1)]

    _, _, H, W = recognition_input_layer.shape
    (real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    annotations = list()
    translated_annotations=list()
    cropped_images = list()
    for i, crop in enumerate(boxes):
        (x_min, y_min, x_max, y_max) = map(int, multiply_by_ratio(ratio_x, ratio_y, crop))
        image_crop                   = run_preprocesing_on_crop(grayscale_image[y_min:y_max, x_min:x_max], (W, H))
        result                       = recognition_compiled_model([image_crop])[recognition_output_layer]
        recognition_results_test     = np.squeeze(result)
        annotation                   = list()
        
        for letter in recognition_results_test:
            parsed_letter = letters[letter.argmax()]
            if parsed_letter == letters[0]:
                break
            annotation.append(parsed_letter)

        real_time_translation(annotation)  

        annotations.append("".join(annotation))
        cropped_image = Image.fromarray(image[y_min:y_max, x_min:x_max])
        cropped_images.append(cropped_image)
        original_text = [annotation for _, annotation in sorted(zip(boxes, annotations), key=lambda x: x[0][0] ** 2 + x[0][1] ** 2)]
        refined_text  = " ".join(original_text)
    boxes_with_annotations = list(zip(boxes, annotations))
    boxes_with_annotations = list(zip(boxes, translated_annotations))
    frame = convert_result_to_image(image, resized_image, boxes_with_annotations, conf_labels=True)

while True:
    _,frame = camera.read()
    
    processing_thread = threading.Thread(target=processing_thread_function, args=(frame,))
    processing_thread.start()
    processing_thread.join()
    
    cv2.imshow("Result",frame)
    key = cv2.waitKey(1)&0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        translation(refined_text)
    elif key == ord('r'):
        play()                
camera.release()
cv2.destroyAllWindows()