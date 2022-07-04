use tch::Tensor;

pub fn main() {
    let model = tch::CModule::load("traced_model.pt").expect("cannot load model");
}
