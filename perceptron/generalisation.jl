
using PyPlot
using Optim

function get_ln_delta(N, d_VC, epsilon)
    return log(4) + d_VC * log(2N) -(epsilon^2)*N/8
end;

d_VC = 10;
epsilon = 0.05;
N_arr = [400000. 420000. 440000. 460000. 480000.];
for N in N_arr
    delta = exp(get_ln_delta(N, d_VC, epsilon));
    println("N = $N, delta = $delta")
end

function get_error(N, d_VC, epsilon, target_delta)
    delta = exp(get_ln_delta(N, d_VC, epsilon));
    return (delta - target_delta)^2;
end;

result = optimize(N -> get_error(N, 10, 0.05, 0.05), 400000, 500000)

N = result.minimum;
err = result.f_minimum;
if result.converged == true
    println("Solution:  N = $N  (error = $err)")
end

function get_vc_bound(N,d_VC,delta)
    inner = log(4) + d_VC * log(2N) - log(delta);
    return sqrt(8inner/N);
end;

function get_rademacher_penalty(N,d_VC,delta)
    inner = 2(log(2N) + d_VC * log(N))/N
    return sqrt(inner) + sqrt(-2log(delta)/N) + 1/N;
end;

function get_parrondo_and_van_den_broek(N,d_VC,delta)
    pvdb_err(epsilon,N,d_VC,delta) = (epsilon - sqrt((2epsilon + log(6) + d_VC*log(2N) - log(delta))/N))^2;
    result = optimize(epsilon -> pvdb_err(epsilon,N,d_VC,delta), 0.0, 10.0);
    if result.converged==true
        return result.minimum
    else
        return NaN
    end;
end;

function get_devroye(N,d_VC,delta)
    devroye_err(epsilon,N,d_VC,delta) = (epsilon - sqrt((4epsilon*(1+epsilon) + log(4) + d_VC*log(N^2) - log(delta))/(2N)))^2;
    result = optimize(epsilon -> devroye_err(epsilon,N,d_VC,delta), 0.0, 20.0);
    if result.converged==true
        return result.minimum
    else
        return NaN
    end;
end;

d_VC = 50;
delta = 0.05;

#N_arr = logspace(0,4,11);  #from 1 to 10k
N_arr = [5, 10, 50, 100, 500, 1000, 5000, 10000];
epsilon_vc = [ get_vc_bound(N,d_VC,delta) for N in N_arr ];
epsilon_rp = [ get_rademacher_penalty(N,d_VC,delta) for N in N_arr ];
epsilon_pv = [ get_parrondo_and_van_den_broek(N,d_VC,delta) for N in N_arr ];
epsilon_dv = [ get_devroye(N,d_VC,delta) for N in N_arr ];

function plot_epsilons(N_arr,epsilon_arr,label_arr)
    fig = figure(figsize=(7,4))
    ax = fig[:add_subplot](1,1,1)
    for i in 1:size(label_arr)[1]
        #ax[:loglog](N_arr,epsilon_arr[:,i],label=label_arr[i]);
        loglog(N_arr,epsilon_arr[:,i],"x-",label=label_arr[i]);
    end;
    #ax[:legend](loc="best",frameon=false);
    legend(loc="best",frameon=false,fontsize=10)
    grid(true);
end;

epsilon_arr = [ epsilon_vc epsilon_rp epsilon_pv epsilon_dv ];
label_arr = [ "VC", "Rademacher", "Parrondo & Van den Broek", "Devroye" ];
plot_epsilons(N_arr,epsilon_arr,label_arr);

println("N = $(N_arr[end])")
println("ϵ (VC)                       = $(epsilon_vc[end])")
println("ϵ (Rademacher)               = $(epsilon_rp[end])")
println("ϵ (Parrondo & Van den Broek) = $(epsilon_pv[end])")
println("ϵ (Devroye)                  = $(epsilon_dv[end])")

println("N = $(N_arr[1])")
println("ϵ (VC)                       = $(epsilon_vc[1])")
println("ϵ (Rademacher)               = $(epsilon_rp[1])")
println("ϵ (Parrondo & Van den Broek) = $(epsilon_pv[1])")
println("ϵ (Devroye)                  = $(epsilon_dv[1])")

function generate_dataset(;n_points=200)
    x = linspace(-1,1,n_points);
    y = sin(π*x);
    x_sample = rand(2);
    y_sample = sin(π*x_sample);
    return x,y,x_sample,y_sample;
end;

function fit_a(x_sample,y_sample)
    sum = x_sample[1]*y_sample[1]+x_sample[2]*y_sample[2]
    return sum/(x_sample[1]^2+x_sample[2]^2);
end;

function predict_y_a(x,a)
    return a .* x;
end;

function plot_xy(x,y,x_sample,y_sample,y_pred)
    figure(figsize=(7,5));
    plot(x,y,"m-",label=L"$\sin(\pi x)$");
    plot(x,y_pred,"g--",label="predicted");
    plot(x_sample,y_sample,"bo",label="sample data");
    grid(true);
    legend(loc="best",frameon=false);
end;
x,y,x_sample,y_sample = generate_dataset();
y_pred = predict_y_a(x,fit_a(x_sample,y_sample));
plot_xy(x,y,x_sample,y_sample,y_pred);

function get_avg_a(n_trials; n_points=200)
    a_arr = Float64[];
    for i in 1:n_trials
        x_sample = rand(2);
        y_sample = sin(π*x_sample);
        a = fit_a(x_sample,y_sample);
        push!(a_arr,a);
    end
    return mean(a_arr);
end;
a = get_avg_a(1e6);    #1.423, 1.425, 1.424, 1.427
println("<g(x)> = $(round(a,2))x");

function get_bias_variance(avg_a, n_trials; n_points=200)
    bias  = Float64[];
    var   = Float64[];
    for i in 1:n_trials
        x_sample = rand(2);
        y_sample = sin(π*x_sample);
        a = fit_a(x_sample,y_sample);
        g_bar_x = avg_a*x_sample;
        push!(bias,mean((g_bar_x - y_sample).^2));
        push!(var,mean((g_bar_x - a .* x_sample).^2));
    end
    return bias, var;
end;

bias,var = get_bias_variance(a, 1e7);
println("<bias> = $(mean(bias))");
println("<var>  = $(mean(var))");
