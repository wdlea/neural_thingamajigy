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

    let (struct_definiton, names) =
        generate_struct_definition(visibility, &name, &num_type, &layers);

    let network_impl = generate_network_impl(num_type, &layers, &names, name);

    let emitted_code = quote! {
        #struct_definiton
        #network_impl
    };

    eprintln!("{}", emitted_code);

    emitted_code.into()
}

fn generate_struct_definition(
    visibility: Visibility,
    name: &Ident,
    num_type: &Type,
    layers: &[LitInt],
) -> (TokenStream, Vec<Ident>) {
    let inputs: Vec<_> = layers.iter().collect();
    let outputs: Vec<_> = layers.iter().skip(1).collect();
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
        names,
    )
}

fn generate_network_impl(
    num_type: Type,
    layers: &[LitInt],
    names: &Vec<Ident>,
    name: Ident,
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
