import torch


def sort_index(teacher_data1,teacher_data2,teacher_data3):
    t1names = [meta["filename"] for meta in teacher_data1["img_metas"]]
    t2names = [meta["filename"] for meta in teacher_data2["img_metas"]]
    t3names = [meta["filename"] for meta in teacher_data3["img_metas"]]

    # [{'file_name':0},]
    idx = [t2names.index(name) for name in t1names]
    teacher_data2["img"]= teacher_data2["img"][
        torch.Tensor(idx)
    ]
    teacher_data2["img_metas"] = [teacher_data2["img_metas"][i] for i in idx]

    '''
    teacher_data1:
    dict{
    
    {
    img:Tensor 5,3,1000,1000
    },
    
    {
    img_metas:
    [{'file_name':'0001.jpg'},{'file_name':'0002.jpg'},{'file_name':'0003.jpg'}]
    }
    
    }
    
    teacher_data2:
    
    dict{
    
    {
    img:Tensor 
    },
    
    {
    img_metas:
    [{'file_name':'0003.jpg'},{'file_name':'0002.jpg'},{'file_name':'0001.jpg'}]
    }
    
    }
        
    teacher_data3:
    
    dict{
    
    {
    img:Tensor
    },
    
    {
    img_metas:
    [{'file_name':'0001.jpg'}]
    }
    
    }
    '''