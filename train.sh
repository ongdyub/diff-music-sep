python train.py experiment=music-separation-16000 model.sde.d_lambda=3.0
python train.py experiment=music-separation-16000 model.sde.d_lambda=2.0
python train.py experiment=music-separation-16000-ouvekh model.sde.sigma_min=0.01 model.sde.sigma_max=0.5 model.sde.theta_min=3.0 model.sde.theta_int_max=3.0 model.sde.theta_rho=0
python train.py experiment=music-separation-16000-ouvekh model.sde.sigma_min=0.01 model.sde.sigma_max=0.5 model.sde.theta_min=2.0 model.sde.theta_int_max=2.0 model.sde.theta_rho=0
python train.py experiment=music-separation-16000-ouvekh model.sde.sigma_min=0.01 model.sde.sigma_max=0.5 model.sde.theta_min=0.1 model.sde.theta_int_max=3.0 model.sde.theta_rho=4
python train.py experiment=music-separation-16000-test