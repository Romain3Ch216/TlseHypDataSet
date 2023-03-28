


transform = transforms.Compose([
                Concat([
                    gabor_transform,
                    compute_spectral_indices
                ]),
                compute_stats])