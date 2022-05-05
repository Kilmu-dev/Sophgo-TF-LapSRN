import bmnett
## compile fp32 model
bmnett.compile(
    model = "/workspace/work/lapsrn/Sophgo-TF-LapSRN/export/LapSRN_x2.pb",     ## Necessary
    outdir = "/fp32bmodel",                    ## Necessary
    target = "BM1684",                ## Necessary
    shapes = [[4,4,1,4]],     ## Necessary
    net_name = "LapSRN",              ## Necessary
    input_names=["IteratorGetNext"],    ## Necessary, when .h5 use None
    output_names=["NCHW_output"], ## Necessary, when .h5 use None
    opt = 1,                           ## optional, if not set, default equal to 1
    dyn = False,                       ## optional, if not set, default equal to False
    cmp = True,                        ## optional, if not set, default equal to True
    enable_profile = True              ## optional, if not set, default equal to False
)