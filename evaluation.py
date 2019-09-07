import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    def compare(start,step,TP,P,T):
        for idx in range(start,len(name_list),step):
            #print('%d/%d'%(idx,len(name_list)))
            name = name_list[idx]
            predict_file = os.path.join(predict_folder,'%s.png'%name)
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        if i%2 != 1:
            print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
        else:
            print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    t_tp = np.mean(np.array(T_TP)[1:])
    p_tp = np.mean(np.array(P_TP)[1:])
    fp_all = np.mean(np.array(FP_ALL)[1:])
    fn_all = np.mean(np.array(FN_ALL)[1:])
    miou_foreground = np.mean(np.array(IoU)[1:])
    print('\n======================================================')
    print('%11s:%7.3f%%'%('mIoU',miou*100))
    print('%11s:%7.3f'%('T/TP',t_tp))
    print('%11s:%7.3f'%('P/TP',p_tp))
    print('%11s:%7.3f'%('FP/ALL',fp_all))
    print('%11s:%7.3f'%('FN/ALL',fn_all))
    print('%11s:%7.3f'%('miou_foreground',miou_foreground))
    loglist['mIoU'] = miou * 100
    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = fp_all
    loglist['fn_all'] = fn_all
    loglist['miou_foreground'] = miou_foreground 
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/val.txt', type=str)
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt',type=str)
    parser.add_argument('--comment', default='', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)
    writelog(args.logfile, loglist, args.comment)
