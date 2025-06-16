use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{parse::Parse, parse_macro_input, Ident, LitInt, Token, Type, Visibility};

mod network_impl;
#[cfg(feature = "train")]
mod random_impl;
#[cfg(feature = "train")]
mod trainable_impl;

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

        while let Ok(width) = input.parse::<LitInt>() {
            let mut reps = 1;
            if input.peek(Token![*]) {
                _ = input.parse::<Token![*]>();

                let count = input.parse::<LitInt>()?;
                reps = count.base10_parse()?;
            }

            for _ in 0..reps {
                hidden.push(width.clone());
            }

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

/// Creates a network with the supplied visibility, name and width arguments.
///
/// ```network!(VISIBILITY NAME, TYPE, GENERATE_TRAINING, WIDTHS);```
/// Params:
///     - `VISIBILITY`, is the visibility *prefix* that the generated network will have. e.g: `pub` or (empty)
///     - `NAME`, is the identifier for the generated network. e.g: `MyNetwork`, `WeatherPredictor`, `Critic`
///     - `TYPE`, is the type of number used in the generated network. Must `impl nalgebra::RealField + Copy`.
///         e.g: `f32` or `f64`
///     - `WIDTHS`, the comma seperated list of layer widths, there are 2 formats: `N`, will
///         produce a single layer with N nodes. `N * M` will produce `M` layers each with `N` nodes.
///         Thus: `5, 5, 5, 6` and `5 * 3, 6` will produce the same network. The first layer width will
///         be the `INPUTS` generic on the produced network wheras the last layer width will be the
///         `OUTPUTS` generic.
///
///
/// # Examples
/// ```rust
///     use neural_thingamajigy::{network, Network, activators::Relu, RandomisableNetwork};
///     
///     // Create a public network with called `EpicName`
///     // with 2 input nodes, 2 hidden layers with 5 nodes each and 1 output node
///     network!(pub EpicName, f32, 2, 5, 5, 1);
///     let my_network = EpicName::random(&mut rand::rngs::OsRng);
///     let output = my_network.evaluate(nalgebra::Vector2::new(1f32, 1f32), &Relu::default());
/// ```
/// ## Shorthand form
/// ```rust
///     # use neural_thingamajigy::network;
///     # use serde;
///     // Both of the below networks have the same structure as the other.
///     // One is a lot easier to read & type.
///     network!(pub CoolName, f32, 5, 5, 5, 6);
///     network!(pub CoolerName, f32, 5 * 3, 6);
///     
/// ```
///
/// # Panics
/// Panics if:
/// - There are less than 2 layer widths
/// - An invalid token is supplied
/// - Never panics at runtime.
#[proc_macro]
pub fn network(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
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
        generate_struct_definition(&visibility, &name, &num_type, &layers);

    let network_impl = network_impl::generate_network_impl(&num_type, &layers, &names, &name);

    #[cfg(feature = "train")]
    let (trainable_network_impl, random_impl) = (
        trainable_impl::generate_trainable_network_impl(
            &visibility,
            &name,
            &names,
            &inputs,
            &outputs,
            &num_type,
        ),
        random_impl::generate_random_impl(&name, &num_type, &names),
    );
    #[cfg(not(feature = "train"))]
    let (trainable_network_impl, random_impl) = (quote! {}, quote! {});

    let emitted_code = quote! {
        #struct_definiton
        #network_impl
        #trainable_network_impl
        #random_impl
    };

    emitted_code.into()
}

fn generate_struct_definition(
    visibility: &Visibility,
    name: &Ident,
    num_type: &Type,
    layers: &[LitInt],
) -> (TokenStream, Vec<LitInt>, Vec<LitInt>, Vec<Ident>) {
    #[cfg(not(feature = "serde"))]
    let serde = quote! {};
    #[cfg(feature = "serde")]
    let serde = quote! {#[derive(serde::Deserialize, serde::Serialize)]};

    let inputs: Vec<_> = layers.to_vec();
    let outputs: Vec<_> = layers.iter().skip(1).cloned().collect();
    let names: Vec<_> = (0usize..)
        .take(outputs.len())
        .map(|i| format_ident!("layer{}", i))
        .collect();

    assert_eq!(inputs.len(), outputs.len() + 1);

    (
        quote! {
            #serde
            #visibility struct #name {
                #(#names: neural_thingamajigy::Layer<#num_type, #inputs, #outputs>), *
            }
        },
        inputs,
        outputs,
        names,
    )
}
