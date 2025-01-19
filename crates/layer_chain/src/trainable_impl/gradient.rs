use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Ident, LitInt, Type};

pub fn generate_trainable_network_gradient(
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

            impl neural_thingamajigy::ValueSet<#num_type> for #inputs_name{
                fn unary_operation(&self, f: impl Fn(&#num_type) -> #num_type) -> Self {
                    Self{
                        #(#names: self.#names.unary_operation(&f)),*
                    }
                }
                fn binary_operation(&self, other: &Self, f: impl Fn(&#num_type, &#num_type) -> #num_type) -> Self {
                    Self{
                        #(#names: self.#names.binary_operation(&other.#names, &f)),*
                    }
                }
                fn unary_inspection(&self, f: &mut impl FnMut(&#num_type)) {
                    #(self.#names.unary_inspection(f);)*
                }
                fn binary_inspection(&self, other: &Self, f: &mut impl FnMut(&#num_type, &#num_type)) {
                    #(self.#names.binary_inspection(&other.#names, f);)*
                }
                fn all(v: #num_type) -> Self{
                    Self{
                        #(#names: neural_thingamajigy::LayerGradient::all(v)),*
                    }
                }
            }
        },
        inputs_name,
    )
}
