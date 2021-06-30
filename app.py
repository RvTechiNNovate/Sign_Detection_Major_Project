from PIL import Image, ImageTk
import tkinter as tk
import cv2

from keras.models import model_from_json
import operator


from string import ascii_uppercase

class Application:
    def __init__(self):
        self.directory = 'E:\\PyAndML\\Project\\ML_project\\Sign-Language-to-Text-master\\model\\'
        # self.hs= SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        self.json_file = open(self.directory+"model-bw.json")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(self.directory+"model-bw.h5")

        self.json_file_dru = open(self.directory+"model-bw_dru.json")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights(self.directory+"model-bw_dru.h5")

        self.json_file_tkdi = open(self.directory+"model-bw_tkdi.json")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights(self.directory+"model-bw_tkdi.h5")

        self.json_file_smn = open(self.directory+"model-bw_smn.json")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights(self.directory+"model-bw_smn.h5")
        
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
          self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.resizable(0,0)
        # self.root.configure(bg="yellow")
        self.Swidth=self.root.winfo_screenwidth()
        self.Sheight=self.root.winfo_screenheight()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.pad=2
        self.root.geometry("{0}x{1}".format(self.Swidth-self.pad,self.Sheight-self.pad))
        self.panel = tk.Label(self.root)
        self.panel.place(x = 107, y = 170, width = 640, height = 500)
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 107, y = 170, width = 640, height = 500)
        
        self.T = tk.Label(self.root)
        self.T.place(relx=0.3,y = 17)
        self.T.config(text = "Sign Language to Text",font=("courier",40,"bold"),bg="cyan")

        self.panel3 = tk.Label(self.root) # Current SYmbol
        self.panel3.place(x = 1050,y=200)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 650,y = 200)
        self.T1.config(text="Character :",font=("Courier",40,"bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 1050,y=300)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 650,y = 300)
        self.T2.config(text ="Word :",font=("Courier",40,"bold"))
        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 1050,y=400)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 650,y = 400)
        self.T3.config(text ="Sentence :",font=("Courier",40,"bold"))



        self.btcall = tk.Button(self.root,command = self.action_call,height = 0,width = 0)
        self.btcall.config(text = "About",font = ("Courier",14))
        self.btcall.place(x = self.root.winfo_screenwidth()-100, y = 5)


        self.str=""
        self.word=""
        self.current_symbol="Empty"
        self.photo="Empty"
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            # print(cv2image.shape)
            cv2image = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2.1)

            ret, res = cv2.threshold(th3,70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            print(res.shape,th3.shape)
            self.predict(th3)
            self.current_image2 = Image.fromarray(th3)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            self.panel3.config(text=self.current_symbol,font=("Courier",50))
            self.panel4.config(text=self.word,font=("Courier",40))
            self.panel5.config(text=self.str,font=("Courier",40))
            
        self.root.after(30, self.video_loop)
    def predict(self,test_image):
        test_image = cv2.resize(test_image, (128,128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))
        prediction={}
        prediction['blank'] = result[0][0]
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1
        #LAYER 1
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        #LAYER 2
        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):
        	prediction = {}
        	prediction['D'] = result_dru[0][0]
        	prediction['R'] = result_dru[0][1]
        	prediction['U'] = result_dru[0][2]
        	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' or self.current_symbol == 'T'):
        	prediction = {}
        	prediction['D'] = result_tkdi[0][0]
        	prediction['I'] = result_tkdi[0][1]
        	prediction['K'] = result_tkdi[0][2]
        	prediction['T'] = result_tkdi[0][3]
        	prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):
        	prediction1 = {}
        	prediction1['M'] = result_smn[0][0]
        	prediction1['N'] = result_smn[0][1]
        	prediction1['S'] = result_smn[0][2]
        	prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
        	if(prediction1[0][0] == 'S'):
        		self.current_symbol = prediction1[0][0]
        	else:
        		self.current_symbol = prediction[0][0]
        if(self.current_symbol == 'blank'):
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.current_symbol] += 1
        if(self.ct[self.current_symbol] > 60):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if(len(self.str) > 16):
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol
    
    
    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
    def destructor1(self):
        print("Closing Application...")
        self.root1.destroy()

    def action_call(self) :
        pass
    #     path="E:\\PyAndML\\Project\\Major_Project\\1.jpg"
    #     self.root1 = tk.Toplevel(self.root)
    #     self.root1.title("Effort By")
    #     image2 =self.Image.open(path)
    #     w = image2.width()
    #     h = image2.height()
    #     self.root1.geometry('%dx%d+0+0' % (w,h))

    #     #labelText.fontsize('10')

    #     label1 = Label(app, image=image2, textvariable=labelText,
    #                 font=("Times New Roman", 24),
    #                 justify=CENTER, height=4, fg="blue")
    #     label1.pack()

        # self.tx = tk.Label(self.root1)
        # self.tx.pack(side='top')
        # self.tx.config(text = "Efforts By", fg="red", font = ("Courier",30,"bold"))

        
        # self.tx6 = tk.Label(self.root1)
        # self.tx6.place(relx=0.1,rely=0.3)
        # self.tx6.config(text = "Ritesh\n17/ICS/118", font = ("Courier",15,"bold"))

       
        # self.tx2 = tk.Label(self.root1)
        # self.tx2.place(relx = 0.3,rely=0.3)
        # self.tx2.config(text = "Anurag\n17/ICS/118", font = ("Courier",15,"bold"))

        
        
        # self.tx3 = tk.Label(self.root1)
        # self.tx3.place(relx = 0.5,rely=0.3)
        # self.tx3.config(text = "Ritesh Srivastav\n17/ICS/118", font = ("Courier",15,"bold"))

      
        # self.tx4 = tk.Label(self.root1)
        # self.tx4.place(relx = 0.7,rely=0.3)
        # self.tx4.config(text = "Arpit\n17/ICS/118", font = ("Courier",15,"bold"))
        
        
        
        # self.tx7 = tk.Label(self.root1)
        # self.tx7.place(relx=0.2,rely=0.4)
        # self.tx7.config(text = "Under the supervision of", fg="red", font = ("Courier",30,"bold"))


        # self.tx6 = tk.Label(self.root1)
        # self.tx6.place(relx = 0.4,rely = 0.5)
        # self.tx6.config(text = "Ankur Sighal Sir", font = ("Courier",30,"bold"))

print("Starting Application...")
pba = Application()
pba.root.mainloop()
