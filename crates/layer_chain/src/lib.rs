use proc_macro::TokenStream;
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
pub fn layer_chain(tokens: TokenStream) -> TokenStream {
    let LayerChainParams {
        visibility,
        num_type,
        name,
        layers,
    } = parse_macro_input!(tokens);
    if layers.len() < 2 {
        panic!("You need to supply at least 2 layer width parameters");
    }

    let inputs: Vec<_> = layers.iter().collect();
    let outputs: Vec<_> = layers.iter().skip(1).collect();
    let names: Vec<_> = (0usize..)
        .take(outputs.len())
        .map(|i| format_ident!("layer{}", i))
        .collect();

    assert_eq!(inputs.len(), outputs.len() + 1);
    assert_eq!(outputs.len(), names.len());

    let struct_definiton = quote! {
        #visibility struct #name {
            #(#names: neural_thingamajigy::Layer<#num_type, #inputs, #outputs>), *
        }
    };

    struct_definiton.into()
}
