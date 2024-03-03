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
        
        plt.plot(train, label='train loss', color='blue')
        plt.plot(validation, label='validation loss', color='orange')
        
        plt.xlabel('epoch')
        plt.ylabel(f'loss ( {loss} )')
        
        # plt.legend()
        
        plt.savefig(f'{dir}/{name}.png')


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
