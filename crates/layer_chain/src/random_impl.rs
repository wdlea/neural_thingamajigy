use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Type};

pub fn generate_random_impl(name: &Ident, num_type: &Type, names: &[Ident]) -> TokenStream {
    quote! {
        impl neural_thingamajigy::RandomisableNetwork<#num_type> for #name{
            fn random(rng: &mut impl Rng) -> Self{
                Self{
                    #(#names: Layer::random(rng)),*
                }
            }
        }
    }
}
