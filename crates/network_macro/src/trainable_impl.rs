use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, LitInt, Type, Visibility};

mod gradient;

pub fn generate_trainable_network_impl(
    visibility: &Visibility,
    name: &Ident,
    names: &[Ident],
    inputs: &[LitInt],
    outputs: &[LitInt],
    num_type: &Type,
) -> TokenStream {
    let (network_layer_inputs_impl, network_layer_inputs_name) =
        generate_trainable_network_inputs(visibility, name, names, inputs, num_type);
    let (network_gradient_impl, network_gradient_impl_name) =
        gradient::generate_trainable_network_gradient(
            visibility, name, names, inputs, outputs, num_type,
        );

    let network_inputs = inputs.first().unwrap();
    let network_outputs = outputs.last().unwrap();

    let evaluate_training_impl =
        generate_evaluate_training_impl(num_type, network_inputs, network_outputs, names);
    let get_gradient_impl =
        generate_get_gradient_impl(num_type, network_inputs, network_outputs, names);
    let apply_nudge_impl = generate_apply_nudge_impl(names);
    quote! {
        #network_layer_inputs_impl
        #network_gradient_impl
        #[cfg(not(target_os = "none"))]
        impl neural_thingamajigy::TrainableNetwork<#num_type, #network_inputs, #network_outputs> for #name{
            type LayerInputs = #network_layer_inputs_name;
            type Gradient = #network_gradient_impl_name;

            #evaluate_training_impl

            #get_gradient_impl

            #apply_nudge_impl
        }
    }
}

fn generate_trainable_network_inputs(
    visibility: &Visibility,
    name: &Ident,
    names: &[Ident],
    inputs: &[LitInt],
    num_type: &Type,
) -> (TokenStream, Ident) {
    let inputs_name = format_ident!("{}LayerInputs", name);

    (
        quote! {
            #[cfg(not(target_os = "none"))]
            #visibility struct #inputs_name{
                #(#names: nalgebra::SVector<#num_type, #inputs>), *
            }
        },
        inputs_name,
    )
}

fn generate_evaluate_training_impl(
    num_type: &Type,
    network_inputs: &LitInt,
    network_outputs: &LitInt,
    names: &[Ident],
) -> TokenStream {
    let input_variable_name: Vec<_> = [format_ident!("inputs")]
        .iter()
        .cloned()
        .chain(names.iter().skip(1).map(|i| format_ident!("{}_input", i)))
        .collect();

    let output_variable_name: Vec<_> = names
        .iter()
        .skip(1)
        .map(|i| format_ident!("{}_input", i))
        .chain([format_ident!("outputs")].iter().cloned())
        .collect();

    quote! {
        fn evaluate_training(
            &self,
            inputs: nalgebra::SVector<#num_type, #network_inputs>,
            activator: &impl neural_thingamajigy::activators::Activator<#num_type>,
        ) -> (nalgebra::SVector<#num_type, #network_outputs>, Self::LayerInputs){
            #(let #output_variable_name = self.#names.through(#input_variable_name, activator);)*

            let all_inputs = Self::LayerInputs{
                #(#names: #input_variable_name),*
            };

            (outputs, all_inputs)
        }
    }
}

fn generate_get_gradient_impl(
    num_type: &Type,
    network_inputs: &LitInt,
    network_outputs: &LitInt,
    names: &[Ident],
) -> TokenStream {
    let reversed_names: Vec<_> = names.iter().rev().collect();

    quote! {
        fn get_gradient(
            &self,
            layer_inputs: &Self::LayerInputs,
            output_loss_gradients: nalgebra::SVector<#num_type, #network_outputs>,
            activator: &impl neural_thingamajigy::activators::Activator<#num_type>,
        ) -> (Self::Gradient, nalgebra::SVector<#num_type, #network_inputs>){
            let current_loss_gradient = output_loss_gradients;
            #(let (#reversed_names, current_loss_gradient) = self.#reversed_names.backpropogate(current_loss_gradient, layer_inputs.#reversed_names, activator);)*

            (
                Self::Gradient{
                    #(#reversed_names),*
                },
                current_loss_gradient
            )
        }
    }
}

fn generate_apply_nudge_impl(names: &[Ident]) -> TokenStream {
    quote! {
        fn apply_nudge(&mut self, nudge: Self::Gradient){
            #(self.#names.apply_shifts(nudge.#names.weight_gradient, nudge.#names.bias_gradient));*
        }
    }
}
