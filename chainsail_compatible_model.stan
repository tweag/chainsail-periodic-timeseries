data{
	int n ;
	vector[n] y ; 
	vector[n] x ;
}
transformed data{
	vector[n] y_scaled = (y-mean(y))/sd(y) ;
}
parameters{
	real<lower=0> noise ;
	real<lower=0> frequency ; 
	// phase & amplitude (but a transform thereof)
	real phamp1 ;
	real phamp2 ;
}
transformed parameters{
        vector[2] phamp = [phamp1, phamp2]' ;
	real amp = sqrt(dot_self(phamp));
	real phase = atan2(phamp[1],phamp[2]);
	vector[n] f = amp * sin(x*frequency - phase) ;
}
model{
	noise ~ weibull(2,1) ; //peaked at ~.8, zero-at-zero, ~2% mass >2
	amp ~ weibull(2,1) ; //ditto
	target += -log(amp) ; // jacobian for setting a prior on amp given it's derivation from phamp
	// phase has an implied uniform prior
	frequency ~ weibull(2, 2) ; // prior on frequency should account for the range and spacing of x
	y_scaled ~ normal(f,noise) ;
}
