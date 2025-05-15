#include <iostream>
#include <ctime>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50

int main() {
    //Eigen 以矩阵为一个基本数据单元，它是一个模板类，它的前三个参数表示：数据类型，行数，列数
    //声明一个 2*3 的 float 矩阵
    Eigen::Matrix<float, 2, 3> matrix_23;
    //同时，Eigen 通过 typedef 提供了许多内置的数据类型，但本质都是 Eigen::Matrix
    //例如 Vector3d 实质上是 Eigen::Matrix<float, 3, 1>
    Eigen::Vector3d v_3d;
    //还有 Matirx3d 实质上是 Eigen::Matrix<double, 3, 1>
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero(); //初始化为0
    //如果不确定矩阵大小，还可以使用动态大小的矩阵
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > matrix_dynamic;
    //动态大小的矩阵还有更简单的写法
    Eigen::MatrixXd matirx_x;

    //矩阵操作
    //输入数据
    matrix_23 << 1, 2, 3, 4, 5, 6;
    //输出
    cout << matrix_23 << endl;

    //用类似于数组下标的形式访问矩阵数据，利用形如(i, j)的坐标形式
    for (int i = 0;i < 1; i++) {
        for (int j = 0; j < 2; j++) {
            cout << matrix_23(i, j) << endl;
        }
    }    

    v_3d << 3, 2, 1;
    //矩阵和向量相乘
    //但是不能混合两种不同类型的矩阵，原因是 eigen 没有在 float 和 double 运算时将对应的变量统一形式的操作
    //需要人为的改变变量形式
    //所以 Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d 就是错的，会报错
    //显式转化方式
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    //当然，矩阵的维数也不能搞错
    //Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;
    //可以看看上述代码的报错

    //一些矩阵运算
    //四维运算直接用对应的运算符即可(+, -, *, /)
    //获得一个随机矩阵
    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl;

    cout << matrix_33.transpose() << endl;//转置
    cout << matrix_33.sum() << endl;//矩阵内给元素之和
    cout << matrix_33.trace() << endl;//矩阵的迹，即主对角线元素之和
    cout << 10 * matrix_33 << endl;//数乘
    cout << matrix_33.inverse() << endl;//矩阵的逆矩阵
    cout << matrix_33.determinant() << endl;//矩阵的行列式


    //特征值与特征向量
    //实对称矩阵可以保证对角化成功
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver ( matrix_33.transpose() * matrix_33 );
    cout << "Eigen value = " << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vector = " << eigen_solver.eigenvectors() << endl;

    // 解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程
    // N 的大小在前边的宏里定义，矩阵由随机数生成
    // 直接求逆自然是最直接的，但是求逆运算量大

    Eigen::Matrix< double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random( MATRIX_SIZE, MATRIX_SIZE );
    Eigen::Matrix< double, MATRIX_SIZE, 1 > v_Nd;
    v_Nd = Eigen::MatrixXd::Random( MATRIX_SIZE, 1 );

    clock_t time_stt = clock();//计时
    //直接求逆
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time use in normal inverse is " << 1000 * (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms" << endl;

    //矩阵分解求法，此处为 QR 分解
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time use in QR compsition is " << 1000 * (clock() - time_stt)/(double)CLOCKS_PER_SEC << "ms" << endl;
    return 0;
}