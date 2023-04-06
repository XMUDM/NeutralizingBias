# NeutralizingBias

### PID

#### training process 

1.  please check these codes are available: 

    1.   app.py  76 line:   trainer.train()

         ```python
         # app.py  76 line
         
             trainer.train()
         ```

    2.  model.py  189 line: self.items = ......

        ```python
        # model.py  189 line
        
                self.items = torch.cat([self.items_id, self.item_pop_true.cuda()], dim=1).to(torch.device('cuda:0'))
        ```

    3.  utils.py  52 line: date_time = 'the unique name for the model' 

        ```python
        # utils.py  52 line
        
                date_time = '2022' 
        ```

2.  please check these codes are not available:

    1.  model.py   184 line - 186 line

        ```python
        # model.py   184 line - 186 line
        
                # item_pop = np.zeros((self.num_items, 1))
                # self.item_pop = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)
                # self.items = torch.cat([self.items_id, self.item_pop.cuda()], dim=1).to(torch.device('cuda:0'))
        ```

    2.  trainer.py  71 line: self.max_epoch ...

        ```python
        # trainer.py  71 line
        
                # self.max_epoch = 48
        ```

3.  run

    ```python
    python -u app.py
    ```



#### testing process 

1.  please check these codes are available: 

    1.  model.py   184 line - 186 line

        ```python
        # model.py   184 line - 186 line
        
                item_pop = np.zeros((self.num_items, 1))
                self.item_pop = Parameter(torch.from_numpy(item_pop).float()).requires_grad_(requires_grad=False)
                self.items = torch.cat([self.items_id, self.item_pop.cuda()], dim=1).to(torch.device('cuda:0'))
        ```

2.  please check these codes are not available:
    1.   app.py  76 line:   trainer.train()

         ```python
         # app.py  76 line
         
             # trainer.train()
         ```

    2.  model.py  189 line: self.items = ......

        ```python
        # model.py  189 line
        
                # self.items = torch.cat([self.items_id, self.item_pop_true.cuda()], dim=1).to(torch.device('cuda:0'))
        ```

3.  please note that you should :
    1.  utils.py  52 line: date_time = 'the unique name for the model' 

        ```python
        # utils.py  52 line
        
                date_time = '2022' 
        ```

    2.  trainer.py  71 line: self.max_epoch = the best epoch

        ```python
        # trainer.py  71 line
        
                self.max_epoch = 48
        ```

4.  run 

    ```python
    python -u app.py
    ```



### IID 

```python
python -u app.py 
```

