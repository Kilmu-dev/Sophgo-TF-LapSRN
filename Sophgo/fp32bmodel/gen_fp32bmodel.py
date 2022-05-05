import bmnett
## compile fp32 model
bmnett.compile(
    model = "../../export/LapSRN_x8.pb",     ## Necessary
    outdir = "/fp32bmodel",                    ## Necessary
    target = "BM1684",                ## Necessary
    shapes = [[3,3,1,64]],     ## Necessary
    net_name = "LapSRN",              ## Necessary
    input_names=["IteratorGetNext:0"],    ## Necessary, when .h5 use None
    output_names=["NCHW_output:0"], ## Necessary, when .h5 use None
    opt = 2,                           ## optional, if not set, default equal to 1
    dyn = False,                       ## optional, if not set, default equal to False
    cmp = True,                        ## optional, if not set, default equal to True
    enable_profile = True              ## optional, if not set, default equal to False
)