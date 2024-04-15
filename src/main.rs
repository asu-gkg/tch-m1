use tch::{Device, kind, Kind, Tensor};


fn main() {
    let shape = [2, 2];
    let b = Tensor::full(shape, 10, (Kind::Uint8, Device::Cpu));
    println!("{}", b);
    let b = b.view((1, 1, -1));
    println!("{}", b);
}
