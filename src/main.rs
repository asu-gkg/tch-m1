use tch::{Device, Kind, Tensor};
fn synthetic_data(w: Tensor, b: Tensor, num_examples: usize) -> (Tensor, Tensor) {
    println!("w.size: {:?}, w.dim: {}", w.size(), w.size()[0]);
    let X = Tensor::randn(&[num_examples as i64, w.size()[0]],
                          (Kind::Float, Device::Cpu));
    println!("X: {}", X);
    let mut y = &X.matmul(&w) + b;
    println!("y: {}", y);
    let mut tmp = y.copy();

    println!("tmp.normal: {}", tmp.normal_(0.0, 0.1));
    let y = &y.copy() + tmp;
    (X, y.reshape([-1, 1]))
}

// 预测房价、线性回归
fn main() {
    let true_w = Tensor::from_slice(&[2.0, -3.4]).to_kind(Kind::Float);
    println!("true_w: {}", true_w);
    let true_b = Tensor::from(3.2);
    let (features, labels) = synthetic_data(true_w, true_b, 10);


    println!("features: {}", features);
    println!("labels: {}", labels);

}
