import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time



class MyUtils:
    
    @staticmethod
    def array_to_color(array):
        
        colors = plt.cm.jet(array)[:, :3]
        
        return colors

    
    @staticmethod
    def plot_loss_progress(train, validation, dir, name='loss_progress', loss='MSE'):
        
        plt.figure(dpi=300)
        plt.plot(train, label='train loss', color='blue')
        plt.plot(validation, label='validation loss', color='orange')
        
        plt.xlabel('epoch')
        plt.ylabel(f'loss ( {loss} )')
        
        plt.ylim(-0.1, max(max(train), max(validation)))
        
        # plt.legend()
        
        plt.savefig(f'{dir}/{name}.png')
    
    
    @staticmethod
    def read_config(config_path):
        
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        config = dict()
        for line in map(lambda x: x.rstrip('\n'), lines):
            key, value = line.split(':')
            config[key.strip()] = value.strip()
        
        return config
    
    
    @staticmethod
    def decide_params(model_name):
        
        params = {
            'length' : 2048,
            'loss' : 'MSE',
            'mode' : 'default',
        }
        
        if model_name == 'PointNet++':
            params['length'] = 1024
            params['mode'] = 'pointnet'
        
        elif model_name == 'SimpleLinear':
            params['length'] = 2048
            params['mode'] = 'color_only'
            
        elif model_name == 'SimpleTransformer':
            params['length'] = 2048
            params['mode'] = 'transformer'
        
        return params


class TimeCounter:

    def __init__(self, length):
        self.length = length
        self.first = time.time()
    
    
    def calc_second(self, index:int):

        try:
            return True, (self.length-index)/index * (time.time() - self.first)
        
        except:
            return False, 0

    
    def predict_time(self, index:int):

        frag, second = self.calc_second(index)
        if not frag: return f'( Time input error. The argment is {index}.)'
        time_stump = self.second_to_time(second)
        
        return time_stump

    
    def second_to_time(self, second:int):

        hour = second//(60*60)
        minite = (second//60) - (hour*60)
        second %= 60
        
        return f'{int(hour):02d}:{int(minite):02d}:{int(second):02d}'
