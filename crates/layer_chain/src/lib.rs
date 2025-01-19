use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse::Parse, parse_macro_input, Ident, LitInt, Token, Type, Visibility};

struct LayerChainParams {
    visibility: Visibility,
    num_type: Type,
    name: Ident,
    layers: Vec<LitInt>,
}

impl Parse for LayerChainParams {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let visibility: Visibility = input.parse()?;

        let name = input.parse()?;
        input.parse::<Token![,]>()?;
        let num_type = input.parse()?;
        input.parse::<Token![,]>()?;

        let mut hidden = Vec::new();

        while let Ok(width) = input.parse() {
            hidden.push(width);

            _ = input.parse::<Token![,]>();
        }

        Ok(Self {
            visibility,
            name,
            num_type,
            layers: hidden,
        })
    }
}

#[proc_macro]
pub fn layer_chain(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let LayerChainParams {
        visibility,
        num_type,
        name,
        layers,
    } = parse_macro_input!(tokens);
    if layers.len() < 2 {
        panic!("You need to supply at least 2 layer width parameters");
    }

    let (struct_definiton, inputs, outputs, names) =
        generate_struct_definition(visibility, &name, &num_type, &layers);

    let network_impl = generate_network_impl(&num_type, &layers, &names, &name);
    let trainable_network_impl =
        generate_trainable_network_impl(&name, &names, &inputs, &outputs, &num_type);

    let emitted_code = quote! {
        #struct_definiton
        #network_impl
        #trainable_network_impl
    };

    eprintln!("{}", emitted_code);

    emitted_code.into()
}

fn generate_struct_definition(
    visibility: Visibility,
    name: &Ident,
    num_type: &Type,
    layers: &[LitInt],
) -> (TokenStream, Vec<LitInt>, Vec<LitInt>, Vec<Ident>) {
    let inputs: Vec<_> = layers.to_vec();
    let outputs: Vec<_> = layers.iter().skip(1).cloned().collect();
    let names: Vec<_> = (0usize..)
        .take(outputs.len())
        .map(|i| format_ident!("layer{}", i))
        .collect();

    assert_eq!(inputs.len(), outputs.len() + 1);

    (
        quote! {
            #visibility struct #name {
                #(#names: neural_thingamajigy::Layer<#num_type, #inputs, #outputs>), *
            }
        },
        inputs,
        outputs,
        names,
    )
}

fn generate_network_impl(
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

fn generate_trainable_network_impl(
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
