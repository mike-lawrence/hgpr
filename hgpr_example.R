# load packages ----
library(tidyverse)
library(rstan)
rstan_options(auto_write = TRUE)

# load ezStan (installing if necessary)
devtools::install_github('mike-lawrence/ezStan')
library(ezStan)
#ezStan has some useful functions for starting & watching parallel chains,
# as well as a nicer summary table of the posterior samples

# Make some fake data ----

# set random seed for reproducibility of data generation
set.seed(1)

# we'll generate pop_data for an experimental design with a single variable
#    "condition" with two levels, but the code generalizes to more variables &
#    levels.

#n_s: number of "subjects" (ex. individual human participants in an experiment)
n_subj = 10

# n_x: number of unique samples on x-axis. Note: as n_x increases, sampling time
#     increases exponentially
n_x = 20

# n_x: number of repeated observations per-x per-condition per-subject
n_reps = 3

# prep a tibble with combination of x, conditions & reps to store the
#    population-level functions
pop_dat = as_tibble(expand.grid(
	x = seq(-10,10,length.out=n_x)
	, rep = 1:n_reps
	, condition = c(-.5,.5)
))

# add some columns, eventually leading to observed pop_data
pop_dat %>%
	dplyr::mutate(
		intercept = sin(x)*dnorm(x,5,8) #a wiggly function
		, effect = (pnorm(x,2,1)-.5)*.1 #a different wiggly function
		, pop_f = intercept + effect*condition
	) ->
	pop_dat

# show the intercept function
pop_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = intercept
		)
	)+
	geom_line()

# show the effect function
pop_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = effect
		)
	)+
	geom_line()


# show the condition functions
pop_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = pop_f
			,  colour = factor(condition)
			,  group = factor(condition)
		)
	)+
	geom_line()

#generate subject functions as deviations from the population functions
subj_dat = purrr::map_df(
	.x = 1:n_subj
	, .f = function(subj_num){
		pop_dat %>%
			dplyr::mutate(
				subj_num = subj_num
				, intercept = intercept + sin((x+rnorm(1,0,2))*rnorm(1,0,.1))/10
				, effect = (pnorm(x,rnorm(1,2,1),rnorm(1,1,.1))-.5)*rnorm(1,.1,.01)
				, subj_f = intercept + effect*condition
				, obs = subj_f + rnorm(n(),0,rweibull(1,2,.1))
			)
	}
)

# show the intercept functions
subj_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = intercept
			, group = subj_num
		)
	)+
	geom_line()


# show the effect functions
subj_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = effect
			, group = subj_num
		)
	)+
	geom_line()

# show the condition functions (overlaid)
subj_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = subj_f
			,  colour = factor(condition)
			,  group = interaction(condition,subj_num)
		)
	)+
	geom_line()


# show the condition functions (faceted)
subj_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = subj_f
			,  colour = factor(condition)
		)
	)+
	geom_line()+
	facet_wrap(~subj_num)

# show the observed data (faceted)
subj_dat %>%
	ggplot(
		mapping = aes(
			x = x
			, y = obs
			, colour = factor(condition)
			, group = interaction(condition,rep)
		)
	)+
	geom_line(alpha=.5)+
	facet_wrap(
		~ subj_num
		, scales='free'
	)

# remove columns we wouldn't actually observe, and turn "condition" into a
#    factor (as is typical in real data), storing the result in "dat"
subj_dat %>%
	dplyr::select(
		x
		, rep
		, condition
		, subj_num
		, obs
	) %>%
	dplyr::mutate(
		condition = factor(dplyr::case_when(
			condition==.5 ~ 'a'
			, condition==-.5 ~ 'b'
		))
	) ->
	dat

# prep the data for modelling ----

# get the sorted unique value for x
x = sort(unique(dat$x))

# for each value in dat$x, get its index x
x_index = match(dat$x,x)

# compute the contrast matrix
z = ezStan::get_contrast_matrix(
	data = dat
	, formula = ~ condition
)
head(z) #show the first bit

# compute the unique entries in the contrast matrix
temp = as.data.frame(z)
temp = tidyr::unite_(data = temp, col = 'combined', from = names(temp))
temp_unique = unique(temp)
z_unique = z[row.names(z)%in%row.names(temp_unique),]
print(z_unique) #show the unique entries in the contrast matrix

# for each row in z, get its index z_unique
z_unique_index = match(temp$combined,temp_unique$combined)

# combine the two index objects to get the index into the flattened z_by_f vector
z_by_f_index = z_unique_index + (x_index-1)*nrow(z_unique)

#get subject indices
subj_inds = ezStan::get_subject_indices(dat$subj_num)
head(subj_inds) #show the first bit
#row for each subject, first column indicating row index corresponding to the
#    beginning of that subject's data, second column indicating row index of the
#    end

# create the data list for stan
data_for_stan = list(
	n_y = nrow(dat)
	, y = scale(dat$obs)[,1] #scaled to mean=0,sd=1
	, n_x = length(x)
	, x = (x-min(x))/(max(x)-min(x)) #scaled to min=0,max=1
	, x_index = x_index
	, n_z = ncol(z)
	, rows_z_unique = nrow(z_unique)
	, z_unique = z_unique
	, z_by_f_index = z_by_f_index
	, n_subj = nrow(subj_inds)
	, subj_inds = subj_inds
)

# Sample the model ----

#compile
hgpr_mod = rstan::stan_model('hgpr.stan')

# start the parallel chains
ezStan::startBigStan(
	stanMod = hgpr_mod
	, stanData = data_for_stan
	, cores = 4 #set this to the # of physical cores on your system
	, iter = 2e3 #2e3 takes about 10min when n_subj=10,n_x=20,n_reps=3
	, stanArgs = "
		include = FALSE
		, pars = c(
			'f_normal01'
			, 'volatility_helper'
			, 'subj_f_normal01'
			, 'subj_volatility_helper'
		)
	"
)

#watch the chains' progress
ezStan::watchBigStan()

#play a sound when done
beepr::beep()

# collect results
post = ezStan::collectBigStan()

# kill just in case
ezStan::killBigStan()

# delete temp folder
ezStan::cleanBigStan()

#how long did it take?
sort(rowSums(get_elapsed_time(post)/60))

#check noise & GP parameters
ezStan::stan_summary(
	from_stan = post
	, par = c('subj_noise_mean','subj_noise_sd','volatility','amplitude')
)

#check the rhats for the population functions
fstats = ezStan::stan_summary(
	from_stan = post
	, par = 'f'
	, return_array = TRUE
)
summary(fstats[,ncol(fstats)]) #rhats


#visualize population functions
f = rstan::extract(
	post
	, pars = 'f'
)[[1]]

f2 = tibble::as_tibble(data.frame(matrix(
	f
	, byrow = F
	, nrow = dim(f)[1]
	, ncol = dim(f)[2]*dim(f)[3]
)))
f2$sample = 1:nrow(f2)


f2 %>%
	tidyr::gather(
		key = 'key'
		, value = 'value'
		, -sample
	) %>%
	dplyr::mutate(
		key = as.numeric(gsub('X','',key))
	) %>%
	dplyr::mutate(
		key = as.numeric(gsub('X','',key))
		, parameter = rep(
			1:dim(f)[3]
			, each = dim(f)[1]*dim(f)[2]
		)
		, x = rep(x,each=dim(f)[1],times=dim(f)[3])
	) %>%
	dplyr::select(
		-key
	) ->
	fdat

fdat %>%
	dplyr::group_by(
		x
		, parameter
	) %>%
	dplyr::summarise(
		med = median(value)
		, lo95 = quantile(value,.025)
		, hi95 = quantile(value,.975)
		, lo50 = quantile(value,.25)
		, hi50 = quantile(value,.75)
	) %>%
	ggplot()+
	geom_hline(yintercept=0)+
	geom_ribbon(
		mapping = aes(
			x = x
			, ymin = lo95
			, ymax = hi95
		)
		, alpha = .5
	)+
	geom_ribbon(
		mapping = aes(
			x = x
			, ymin = lo50
			, ymax = hi50
		)
		, alpha = .5
	)+
	geom_line(
		mapping = aes(
			x = x
			, y = med
		)
		, alpha = .5
	)+
	facet_grid(
		parameter ~ .
		, scale = 'free_y'
	)
