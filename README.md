# NeutralizingBias



1.  PID

    1.  training process 

        1.  please check these codes are available: 

            1.   app.py  76 line:   trainer.train()
            2.  model.py  189 line: self.items = ......

        2.  please check these codes are not available:

            1.  model.py   184 line - 186 line
            2.  trainer.py  171 line: self.max_epoch ...

            

    2.  testing process 

        1.  please check these codes are available: 
            1.  model.py   184 line - 186 line
        2.  please check these codes are not available:
            1.   app.py  76 line:   trainer.train()
            2.  model.py  189 line: self.items = ......
        3.  please note that you should :
            1.  utils.py  52 line: date_time = 'the time you train the model' 
            2.  trainer.py  171 line: self.max_epoch = the best epoch