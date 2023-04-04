from bvae import OptimizedBVAE
from dataloader import dataset_list, Dataloader

z_dim = 2

for norm_f in ["l1", "l2"]:
    dataset_list.reverse()
    for ds in dataset_list:
        for mode in ["feat_agg", "feat_raw", "llm_all-MiniLM-L6-v2"]:
            dl = Dataloader(
                    datasetname=ds,
                    norm_f=norm_f,
                    mode=mode,
                    )
            df_answ, df_text = dl.df_answ, dl.df_text
            print(f"{norm_f} {ds} {mode}")

            features = df_answ.values.shape[1]

            # Define Optimized BVAE
            optimized_bvae = OptimizedBVAE(
                    input_dim=features,
                    z_dim=z_dim,
                    dataset_size=len(df_answ),
                    variational=True,
                    use_VeLO=False,
                    verbose=True,
                    )

            # Fit Optimized BVAE
            model = optimized_bvae.fit(df_answ)
            breakpoint()
