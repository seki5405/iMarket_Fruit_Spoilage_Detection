# iMarket_README

![Untitled](iMarket_README%20a587e9ae02cb4b6f8b298244276c7d67/Untitled.png)
 
iMarket is a Fruit freshness evaluation tools based on YOLOv5 and Pretrained vision network. It’s forked from [Ultralytics](https://ultralytics.com/)’ YOLOv5 repository and based on this, added and customized for our purpose.

### How to use

---

- Install
    
    Clone repo and install requirements.txt in a Python≥3.7.0 environment, including PyTorch≥1.7.
    
    ```python
    git clone https://github.com/seki5405/iMarket_Fruit_Spoilage_Detection.git
    cd iMarket_Fruit_Spoilage_Detection
    pip install -r requirements.txt
    ```
    

- Training
    - Train regression model
        - `freshness_train.py` is for training your regression model
        - You have to train for each fruit with its own classified dataset
        - There are commented out codes for classification approach(#classfication)
        
        ```python
        python3 freshness_train.py --save-name 'pretrained weights path'\
        													  --epochs 50 --dataset 'dataset path'
        /* Parser information
        '--base-model', type=str, default='vgg16', help='Base model for the regression model'
        '--epochs', type=int, default=50, help='Training epochs'
        '--batch-size', type=int, default=32, help='Training batch size'
        '--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer'
        '--save-name', type=str, required=True, help='Name to save weights after training'
        '--dataset', type=str, required=True, help='Dataset path'
        '--imgsz', '--img', '--img-size', type=int, default=416, help='Image size (width = height)'
        '--split', type=float, default=0.2, help='train_valid split ratio' */
        ```
        
    - Train YOLOv5 based model
        - `yolo_train.py` is for training Object Detection model
        - It’s mainly inherited from YOLOv5 except some customized functions
        
        ```python
        !python yolo_train.py --img 416 --epochs 400 --batch 32 \
        										  --data 'yaml_path' --cfg models/yolov5s.yaml  \
        											--weights yolov5s.pt --name 'save name' \
        // You can change the arguments and add above this
        ```
        
- Prediction
    
    `imarket_main.py` is the main function to implement evaluation for each fruits
    
    - To visualize the results on colab, use the cod below
        
        ```python
        // For visualizaiton in colab, use this code
        import cv2
        from google.colab.patches import cv2_imshow
        
        def show_img(url):
          img_name = url.split('/')[-1]
          img_path = os.path.join("saved dir")
          for path, dir, fname in os.walk('saved dir'):
            if img_name in fname:
              f_path = os.path.join(path, img_name)
          img = cv2.imread(f_path)
          cv2_imshow(img)
        ```
        
    - To implement the main function
        
        ```python
        URL = "Your own image url"
        python imarket_main.py --weights $yolo_path --freshness-weights $reg_path --imgsz 416 --conf 0.25 --source $URL
        show_img(URL)
        ```
        
    - Example of result
        
        
        ![Untitled](iMarket_README%20a587e9ae02cb4b6f8b298244276c7d67/Untitled%201.png)
        
        ![Untitled](iMarket_README%20a587e9ae02cb4b6f8b298244276c7d67/Untitled%202.png)