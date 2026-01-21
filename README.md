# Python-Face-Detection-using-LBPH-model
Using python to detect faces by LBPH and Haar Cascade algorithm. A project I made for my University Course (Digital Signal Processing Sessional)

**Programs needed:**
1.	VS Code
2.	Python (recommended version 3.10)
3.	Camera (you can use the built in camera of Laptop or use an external software like Camo Studio connecting your Phone and PC)



**Descriptions with steps:**
1.	You should always try configuring with **data_collection.py**
2.	The code should run without any problems with the correct python libraries. If any problems specially with opencv, try using 
Open CV: pip install opencv-contrib-python 
3.	After running the data_collection.py, it will ask for argument namely which user you want to configure, how many pictures you want to take etc. Bash the following code in the Terminal

**python data_collection.py –user “USER NAME” –cam 0 –per-pose 125**


There are also more commands that can be changed to make running the project with your control.
Simply add (“—interval 0.5, --min-size 80,”) Check out more details in the following section

```markdown
<details>
<summary><strong>Main – Capture + Auto-train</strong></summary>

```python
def main():
    parser = argparse.ArgumentParser(
        description="Offline face data collection (tracking) + optional full-frame saves + auto-train LBPH."
    )

    parser.add_argument("--user", required=True, help="User label (folder name)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--per-pose", type=int, default=25, help="Images to capture")
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between saves")
    parser.add_argument("--min-size", type=int, default=90, help="Min face size (px)")
    parser.add_argument("--blur-thresh", type=float, default=18.0, help="Min Laplacian variance")
    parser.add_argument("--brightness-min", type=float, default=45.0, help="Min brightness")
    parser.add_argument("--tracker", choices=["none", "csrt", "kcf"], default="csrt", help="Stabilize bounding box")
    parser.add_argument("--bypass-quality", action="store_true", help="Save even if quality checks fail")
    parser.add_argument("--no-train", action="store_true", help="Do not train after capture (default trains)")
```


    
4.	The camera will popup and will take pictures of you based on the parameters set. By default, it takes 125 pictures. 
5.	After collection of pictures is completed, the LBPH will train and store the data in “model” folder, in an yml and json format. 
6.	Now use the pic_password.py code and run the code.
7.	The terminal will again ask for argument and bash the following code in the terminal
 
python pic_password.py --user "USER NAME" --cam 0


9.	A new camera will popup and should detect your face with an average accuracy of above 60%. With better light conditions and less changes to poses, the accuracy should increase.
Applications of this project: 
The goal of this project was to implement in an offline, low-cost hardware environment for example: cheap face detection systems in universitites.
**If any problem arises, feel free to contact me :)**
