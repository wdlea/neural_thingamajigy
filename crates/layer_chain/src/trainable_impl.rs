use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, LitInt, Type};

pub fn generate_trainable_network_impl(
    name: &Ident,
    names: &[Ident],
    inputs: &[LitInt],
    outputs: &[LitInt],
    num_type: &Type,
) -> TokenStream {
    let (network_layer_inputs_impl, network_layer_inputs_name) =
        generate_trainable_network_inputs(name, names, inputs, num_type);
    let (network_gradient_impl, network_gradient_impl_name) =
        generate_trainable_network_gradient(name, names, inputs, outputs, num_type);

    let network_inputs = inputs.first().unwrap();
    let network_outputs = outputs.last().unwrap();

    quote! {
        #network_layer_inputs_impl
        #network_gradient_impl
        impl neural_thingamajigy::TrainableNetwork<#num_type, #network_inputs, #network_outputs> for #name{
            type LayerInputs = #network_layer_inputs_name;
            type Gradient = #network_gradient_impl_name;

            fn evaluate_training(
                &self,
                inputs: nalgebra::SVector<#num_type, #network_inputs>,
                activator: &impl neural_thingamajigy::activators::Activator<#num_type>,
            ) -> (nalgebra::SVector<#num_type, #network_outputs>, Self::LayerInputs){
                todo!();
            }

            fn get_gradient(
                &self,
                layer_inputs: &Self::LayerInputs,
                output_loss_gradients: nalgebra::SVector<#num_type, #network_outputs>,
                activator: &impl neural_thingamajigy::activators::Activator<#num_type>,
            ) -> (Self::Gradient, nalgebra::SVector<#num_type, #network_inputs>){
                todo!();
            }

            fn apply_nudge(&mut self, nudge: Self::Gradient){
                todo!();
            }
        }
    }
}

fn generate_trainable_network_inputs(
    name: &Ident,
    names: &[Ident],
    inputs: &[LitInt],
    num_type: &Type,
) -> (TokenStream, Ident) {
    let inputs_name = format_ident!("{}LayerInputs", name);

    (
        quote! {
            struct #inputs_name{
                #(#names: nalgebra::SVector<#num_type, #inputs>), *
            }
        },
        inputs_name,
    )
}

fn generate_trainable_network_gradient(
    name: &Ident,
    names: &[Ident],
    inputs: &[LitInt],
    outputs: &[LitInt],
    num_type: &Type,
) -> (TokenStream, Ident) {
    let inputs_name = format_ident!("{}Gradient", name);

    (
        quote! {
            #[derive(Default)]
            struct #inputs_name{
                #(#names: neural_thingamajigy::LayerGradient<#num_type, #inputs, #outputs>), *
            }
        },
        inputs_name,
    )
}
