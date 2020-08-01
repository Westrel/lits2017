'''路径'''
# 原始路径
train_ct_path = 'D:\\WangXuan\\python\\deeplearning\\lits2017\\dataset\\CT\\'
train_seg_path = 'D:\\WangXuan\\python\\deeplearning\\lits2017\\dataset\\SEG\\'
test_ct_path = 'D:\\WangXuan\\python\\deeplearning\\lits2017\\test\\CT'
test_seg_path = 'D:\\WangXuan\\python\\deeplearning\\lits2017\\test\\SEG'
range_ = '3'
# 训练路径
training_set_path = './train/'+range_+'/'
pred_path = './test_pred/'+range_+'/'
# pred_path = './pred/'
crf_path = './test_crf/' # crf优化结果
# crf_path = './crf/'
module_path = './module/'+range_+'/net950-0.021-0.106.pth'
save_module_path = './module/'+range_+'/'
result_path = './result/'+range_+'.xls'


'''数据获取'''
# 使用48张连续切片作为网络的输入
size = 24
# 横断面降采样因子
down_scale = 0.5
# 仅使用包含肝脏以及肝脏上下20站切片作为训练样本
expand_slice = 40
# 将所有数据在z轴的spacing归一化到1mm
slice_thickness = 1
# CT数据灰度截断窗口
upper, lower = 200, -200


'''网络结构'''
# dropout随机丢弃概率
drop_rate = 0.3


'''网络训练'''
# 显卡序号
gpu = '0'

Epoch = 5000

learning_rate = 1e-4

learning_rate_decay = [500, 750]
# 深度监督衰减系数
alpha = 0.33

batch_size = 1

num_workers = 8

pin_memory = True

cudnn_benchmark = True


'''模型测试'''
# 阈值
threshold = 0.5
# 滑动取样步长
stride = 12
# 最大空洞面积
maximum_hole = 5e4

'''CRF后处理优化相关参数'''
# 根据预测结果在三个方向上的扩展数量
z_expand, x_expand, y_expand = 10, 30, 30
# CRF迭代次数
max_iter = 20
# CRF高斯核函数

# 恢复训练
recovery = True