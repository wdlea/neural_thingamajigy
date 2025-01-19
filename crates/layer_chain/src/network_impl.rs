use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, LitInt, Type};

pub fn generate_network_impl(
    num_type: &Type,
    layers: &[LitInt],
    names: &Vec<Ident>,
    name: &Ident,
) -> TokenStream {
    let network_inputs = layers.first().unwrap();
    let network_outputs = layers.last().unwrap();

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
        impl neural_thingamajigy::Network<#num_type, #network_inputs, #network_outputs> for #name{
            fn evaluate(
                &self,
                inputs: nalgebra::SVector<#num_type, #network_inputs>,
                activator: &impl neural_thingamajigy::activators::Activator<#num_type>,
            ) -> nalgebra::SVector<#num_type, #network_outputs>{
                #(let #output_variable_name = self.#names.through(#input_variable_name, activator);)*

                outputs
            }
        }
    }
}
