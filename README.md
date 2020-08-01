# lits2017肝脏分割

[项目源码地址](https://github.com/assassint2017/MICCAI-LITS2017)

做了些许结构上的修改

运行前在parameter.py中修改相关参数

然后运行：

1. get_training_set.py 获取数据集
2. train_ds.py 训练
3. val.py 验证
4. 可以使用show.py查看nii图像

**注**

show.py用法：可以在其代码里修改main()函数中的路径，也可以在命令行中使用

```cmd
python show.py path
```

来查看

