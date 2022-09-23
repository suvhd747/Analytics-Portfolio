# apPETite-package
Home to our custom and fully customizable package that can be used to generate new protein or enzyme sequences!

## Package Description and Functions

### construct_diwv()
- This function returns a dictionary that maps pairs of amino acids to an instability score based on the ProtParam Instability Index. 
- The index is included, but you are welcome to use any other index.

### passes_filters()
- This function creates filters that will be used on the generated novel protein sequences and let those that satisfy the filter’s specifications return as a valid sequence. This function currently has filters for sequence length and sequence stability and instability.
- These filters can be modified accordingly for your protein or enzyme, and you can also add your own filters for any other property that you are interested in analyzing.

### run_rnn()
- This is the main function that takes an input of training sequences, and will return new sequences based on the specified filters from passes_filters(). 
- This function has the most adaptability, as you can adjust any of the parameters involved here to whatever you’d like. The ones we used specifically for our PETase work are included here.