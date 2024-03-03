import numpy as np
import osada
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)



## =====================================================================
## 


class Sensor:
    
    
    ## =====================================================================
    ## 
    
    
    def __init__(self, radius=None, led_deg=None, pt_deg=None, sensor_preset='classic',
                 ref_deg=np.array([0., 0.]), led_length=16, pt_length=25) -> None:
        
        
        osada.cprint('@ sensor initialization', 'orange')
        
        
        ## 回転後の軸の向き
        self.element_vecs = [[np.zeros(3), 1] for _ in range(3)]
        
        
        ## =====================================================================
        ## 定数
        
        
        ## 光源の強度 [lx]
        ## elem, 2e-8 : 目標L_0=5未満
        self.L_0_ELEM = 2e-8
        self.L_0 = self.L_0_ELEM * 1.
        ## 電気素量 [C]
        self.ELEMENTARY_CHARGE = 1.602176634e-19
        ## 量子効率
        self.QUANTUM_EFFICIENCY = 1.
        ## プランク定数 [J s]
        self.PLANCK_CONSTANT = 6.62607015e-34
        ## 光の波長 [m]
        self.LAMBDA = 850e-9
        ## 光の周波数 [Hz]
        self.FREQUENCY = 1. / self.LAMBDA
        ## ベース接地電流増幅率 (\alpha)
        self.AMPLIFICATION_ALPHA = 0.9
        ## 電流増幅率 (\beta)
        self.AMPLIFICATION_BETA = 1 / (1 - self.AMPLIFICATION_ALPHA)
        
        ## C_{LED} : LEDの強度
        self.C_LED = np.log(0.5) / np.log(np.cos(np.pi / 3))
        ## C_{PT} : フォトトランジスタの感度
        self.C_PT = np.log(0.35) / np.log(np.cos(np.pi / 18))
        
        
        ## =====================================================================
        ## LED・受光部
        
        
        ## 初期の向き
        self.INIT_DIRECTION = np.array([1., 0., 0.])
        ## 初期の上方向
        self.INIT_UPWARD = np.array([0., 0., 1.])
        ## 計算不能の時に代わりに使う回転軸（INITを90度回転したもの）
        self.ALT_ROTATE_DIRECTION = np.array([0., 0., 1.])
        
        ## 基準点
        self.ref_deg = ref_deg
        self.ref_point = np.array([0., 0., 0.])
        ## センサの中心
        self.center = np.array([0., 0., 0.])
        
        
        self.sensor_preset = sensor_preset
        
        
        ## 半径と座標
        if led_deg is None or pt_deg is None:            
            
            
            ## 旧センサ
            if sensor_preset == 'classic':
                
                self.radius = 35. / 2.
                
                self.led_deg = np.asarray([
                            [-30.,  30.], [-10.,  30.], [10.,  30.], [30.,  30.],
                            [-30.,  10.], [-10.,  10.], [10.,  10.], [30.,  10.],
                            [-30., -10.], [-10., -10.], [10., -10.], [30., -10.],
                            [-30., -30.], [-10., -30.], [10., -30.], [30., -30.],
                        ], dtype=np.float32)
                self.pt_deg = np.asarray([
                            [-40.,  40.], [-20.,  40.], [0.,  40.], [20.,  40.], [40.,  40.],
                            [-40.,  20.], [-20.,  20.], [0.,  20.], [20.,  20.], [40.,  20.],
                            [-40.,  0. ], [-20.,   0.], [0.,   0.], [20.,   0.], [40.,   0.],
                            [-40., -20.], [-20., -20.], [0., -20.], [20., -20.], [40., -20.],
                            [-40., -40.], [-20., -40.], [0., -40.], [20., -40.], [40., -40.],
                        ], dtype=np.float32)
            
                self.led_length = 16.
                self.pt_length = 25.

            
            ## 新センサ
            elif sensor_preset == 'novel':
                
                self.radius = 35. / 2.
                
                self.led_deg = np.asarray([
                            [-50.,  50.], [-30.,  50.], [-10.,  50.], [10.,  50.], [30.,  50.], [50.,  50.],
                            [-50.,  30.], [-30.,  30.], [-10.,  30.], [10.,  30.], [30.,  30.], [50.,  30.],
                            [-50.,  10.], [-30.,  10.], [-10.,  10.], [10.,  10.], [30.,  10.], [50.,  10.],
                            [-50., -10.], [-30., -10.], [-10., -10.], [10., -10.], [30., -10.], [50., -10.],
                            [-50., -30.], [-30., -30.], [-10., -30.], [10., -30.], [30., -30.], [50., -30.],
                            [-50., -50.], [-30., -50.], [-10., -50.], [10., -50.], [30., -50.], [50., -50.],
                        ], dtype=np.float32)
                self.pt_deg = np.asarray([
                            [-40.,  40.], [-20.,  40.], [0.,  40.], [20.,  40.], [40.,  40.],
                            [-40.,  20.], [-20.,  20.], [0.,  20.], [20.,  20.], [40.,  20.],
                            [-40.,  0. ], [-20.,   0.], [0.,   0.], [20.,   0.], [40.,   0.],
                            [-40., -20.], [-20., -20.], [0., -20.], [20., -20.], [40., -20.],
                            [-40., -40.], [-20., -40.], [0., -40.], [20., -40.], [40., -40.],
                        ], dtype=np.float32)
                
                self.led_length = 36.
                self.pt_length = 25.


            ## 新センサ（ToF）
            elif sensor_preset == 'novel ToF':
                
                self.radius = 35. / 2.
                
                self.led_deg = np.asarray([
                            [-50.,  50.], [-30.,  50.], [-10.,  50.], [10.,  50.], [30.,  50.], [50.,  50.],
                            [-50.,  30.], [-30.,  30.], [-10.,  30.], [10.,  30.], [30.,  30.], [50.,  30.],
                            [-50.,  10.], [-30.,  10.], [-10.,  10.], [10.,  10.], [30.,  10.], [50.,  10.],
                            [-50., -10.], [-30., -10.], [-10., -10.], [10., -10.], [30., -10.], [50., -10.],
                            [-50., -30.], [-30., -30.], [-10., -30.], [10., -30.], [30., -30.], [50., -30.],
                            [-50., -50.], [-30., -50.], [-10., -50.], [10., -50.], [30., -50.], [50., -50.],
                        ], dtype=np.float32)
                self.pt_deg = np.asarray([
                            [-40.,  40.], [-20.,  40.], [0.,  40.], [20.,  40.], [40.,  40.],
                            [-40.,  20.], [-20.,  20.], [0.,  20.], [20.,  20.], [40.,  20.],
                            [-40.,  0. ], [-20.,   0.],             [20.,   0.], [40.,   0.],
                            [-40., -20.], [-20., -20.], [0., -20.], [20., -20.], [40., -20.],
                            [-40., -40.], [-20., -40.], [0., -40.], [20., -40.], [40., -40.],
                        ], dtype=np.float32)
                
                self.led_length = 36.
                self.pt_length = 24.


            else:
                raise ValueError('sensor_preset must be "classic" or "novel".')
            
            
        else:
            
            self.sensor_preset = 'custom'
            
            self.radius = radius
            
            self.led_deg = led_deg
            self.pt_deg = pt_deg
            
            self.led_length = float(led_length)
            self.pt_length = float(pt_length)

        
        osada.cprint(f'  | preset: {self.sensor_preset}', 'yellow')
        
        
        ## =====================================================================
        ## 初期の処理
        
        
        if radius is not None:
        
            self.rotate_from_direction(first_step=True)
            self.translate()
    
    
    ## eη/hν
    def get_coefficient_P2I(self):
        
        return (self.ELEMENTARY_CHARGE * self.QUANTUM_EFFICIENCY) / (self.PLANCK_CONSTANT * self.FREQUENCY)
    
        
    ## =====================================================================
    ## 角度に従う特性
    
    
    def H_LED(self, theta) -> float:
        
        ## H_{LED}(theta) = cos^C_{LED}(theta)
        result = np.power(np.cos(theta), self.C_LED)
        
        if np.isnan(result): return 0
        return max(result, 0)


    def H_PT(self, theta) -> float:
        
        ## H_{PT}(theta) = cos^C_{PT}(theta)
        result = np.power(np.cos(theta), self.C_PT)
        
        if np.isnan(result): return 0
        return max(result, 0)
    
    
    def cos_tyler(theta):
        
        approx = 1 - np.power(theta, 2) / 2
        
        return approx
    
    
    ## =====================================================================
    ## 回転
    
    
    def rotate_vector(self, v, axis, angle):
        '''
        @func : 回転行列
        '''
        
        ## 外積が0ベクトルの場合、回転軸を規定のものに変更
        if np.linalg.norm(axis) == 0:
            axis = np.cross(self.INIT_DIRECTION, self.ALT_ROTATE_DIRECTION)
        
        # quaternion = self.quaternion_from_axis_angle(axis, angle)
        # return self.quaternion_rotate_vector(quaternion, v)
        
        axis /= np.linalg.norm(axis)
        axis_x, axis_y, axis_z = axis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        ## ロドリゲスの回転公式
        rotation_matrix = np.array([
            [axis_x**2 * (1 - cos_angle) + cos_angle,
            axis_x * axis_y * (1 - cos_angle) - axis_z * sin_angle,
            axis_x * axis_z * (1 - cos_angle) + axis_y * sin_angle],

            [axis_y * axis_x * (1 - cos_angle) + axis_z * sin_angle,
            axis_y**2 * (1 - cos_angle) + cos_angle,
            axis_y * axis_z * (1 - cos_angle) - axis_x * sin_angle],

            [axis_z * axis_x * (1 - cos_angle) - axis_y * sin_angle,
            axis_z * axis_y * (1 - cos_angle) + axis_x * sin_angle,
            axis_z**2 * (1 - cos_angle) + cos_angle]
        ])
        
        rotated = np.dot(rotation_matrix, v)
        
        return rotated
    
    
    
    def rotate_from_direction(self, direction:np.ndarray=None, upward:np.ndarray=None, first_step:bool=False) -> None:
        '''
        @func : LED/PTの座標を代表点の方向から計算
        '''
        
        if direction is None:
            direction = self.INIT_DIRECTION.copy()
        else:
            direction = np.asarray(direction, dtype=np.float32)
        
        if upward is None:
            upward = self.INIT_UPWARD.copy()
        else:
            upward = np.asarray(upward, dtype=np.float32)
         
        direction /= np.linalg.norm(direction)
        self.direction = direction.copy()
        
        
        ## 外積より回転軸
        axis = np.cross(self.INIT_DIRECTION, direction)
        
        ## 内積より回転角
        angle = np.arccos(np.dot(self.INIT_DIRECTION, direction))
        
        ## 回転後の上方向
        rotated_upward = self.rotate_vector(self.INIT_UPWARD, axis, angle)

        self.element_vecs[0] = [upward.copy(), 1]
        self.element_vecs[1] = [direction.copy(), 1]
        self.element_vecs[2] = [np.cross(direction, upward), 1]
        
        upward_cos = np.dot(rotated_upward, upward)
        
        ## -1に近い場合、π回転
        if np.abs(upward_cos + 1) < 1e-5:
            upward_axis = direction.copy()
            upward_angle = np.pi
        ## それ以外
        else:
            upward_axis = np.cross(rotated_upward, upward)
            upward_angle = np.arccos(upward_cos)
            
        ## 1に近い場合、回転なし
        if np.isnan(upward_angle):
            upward_axis = direction.copy()
            upward_angle = 0
            
        ## =====================================================================
        ## LED
        
        rad_xy = np.radians(self.led_deg)
        xyz = np.zeros((len(self.led_deg), 3))        
        
        for i, rad in enumerate(rad_xy):
            
            x_r, y_r = rad
            
            x = self.radius * np.cos(x_r) * np.cos(y_r)
            y = self.radius * np.sin(x_r) * np.cos(y_r)
            z = self.radius * np.sin(y_r)

            rotated = self.rotate_vector(np.array([x, y, z]), axis, angle)
            rotated = self.rotate_vector(rotated, upward_axis, upward_angle)

            xyz[i] += rotated
        
        self.led_xyz = xyz.copy()
        del xyz, rad_xy
        
        ## =====================================================================
        ## PT
        
        rad_xy = np.radians(self.pt_deg)
        xyz = np.zeros((len(self.pt_deg), 3))
        
        for i, rad in enumerate(rad_xy):
            
            x_r, y_r = rad
            
            x = self.radius * np.cos(x_r) * np.cos(y_r)
            y = self.radius * np.sin(x_r) * np.cos(y_r)
            z = self.radius * np.sin(y_r)

            rotated = self.rotate_vector(np.array([x, y, z]), axis, angle)         
            rotated = self.rotate_vector(rotated, upward_axis, upward_angle)
    
            if x_r == self.ref_deg[0] and y_r == self.ref_deg[1]:
                self.ref_point = rotated.copy()

            xyz[i] += rotated
        
        self.pt_xyz = xyz.copy()
        del xyz, rad_xy
        
        
        return
    
    
    
    ## =====================================================================
    ## 並進
    
    
    def translate(self, delta:list=[0., 0., 0.]) -> None:
        
        self.center = np.asarray(delta, dtype=np.float32)
        
        return
    
    
    ## =====================================================================
    ## xyzを取得
    
    
    def get_led_xyz(self, ) -> np.ndarray:
        
        return self.led_xyz + self.center - self.ref_point
    
    
    def get_pt_xyz(self, ) -> np.ndarray:
        
        return self.pt_xyz + self.center - self.ref_point
    
    
    def get_ref_point(self, ) -> np.ndarray:
        
        return self.center
    
    
    ## =====================================================================
    ## 法線ベクトルを取得
    
    
    def get_led_normals(self, ) -> np.ndarray:
        
        normals = [
            n/np.linalg.norm(n) if np.linalg.norm(n)!=0 else n for n in self.led_xyz
            ]
        
        return normals
    
    
    def get_pt_normals(self, ) -> np.ndarray:
        
        normals = [
            n/np.linalg.norm(n) if np.linalg.norm(n)!=0 else n for n in self.pt_xyz
            ]
        
        return normals
    
    
    def get_ref_normal(self, ) -> np.ndarray:
        
        if np.linalg.norm(self.ref_point) == 0.:
            return self.ref_point
        return self.ref_point / np.linalg.norm(self.ref_point)
    
    
    ## =====================================================================
    ## 球の中心座標を取得
    
    
    def get_center(self, ) -> np.ndarray:
        
        return self.center - self.ref_point
    
    
    def get_sensor_direction(self, ) -> np.ndarray:
            
        return self.direction            
    
    
    ## =====================================================================
    ## Debug
    

    def debug(self, save_dir='.'):
        
        import matplotlib.pyplot as plt
        import imageio
        import os
        
        images = []
        length = 50
        
        fig = plt.figure(figsize=(10, 10), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        
        print()
        for i, deg in enumerate(np.linspace(0, 2*np.pi, length)):
            
            x_direct, y_direct = np.cos(deg), np.sin(deg)
            
            self.rotate_from_direction([x_direct, y_direct, 0])
            self.translate()
            
            x, y, z = zip(*self.get_led_xyz())
            ax.scatter(x, y, z, color='r', label='IR LED', s=50)        
            normals = self.get_led_normals()
            for point, normal in zip(self.get_led_xyz(), normals):
                ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=5, normalize=True)

            x, y, z = zip(*self.get_pt_xyz())
            ax.scatter(x, y, z, color='b', label='Photo Transistor', s=50)
            normals = self.get_pt_normals()
            for point, normal in zip(self.get_pt_xyz(), normals):
                ax.quiver(point[0], point[1], point[2], normal[0], normal[1], normal[2], length=5, normalize=True)
                

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_zlim(-15, 15)
            ax.set_box_aspect([1,1,1])

            ax.set_position([0, 0, 1, 1])

            plt.savefig(f'{save_dir}/debug.png')
            plt.clf()

            images.append(imageio.imread(f'{save_dir}/debug.png'))
            os.remove(f'{save_dir}/debug.png')
            
            print(f'\033[1A{i+1} / {length}')
        
        
        imageio.mimsave(f'{save_dir}/sensor_coordinates.gif', images, fps=length / 5, loop=0)
            
        