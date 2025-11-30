import torch
import os

def load_real_image(folder = "data/", img_name = None, idx = 0, img_size=512, device='cuda'):
    from ddm_inversion.utils import pil_to_tensor
    from PIL import Image
    from glob import glob
    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size,
                                    img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1]== 4:
        img = img[:,:3,:,:]
    return img

def mu_tilde(model, xt,x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t 
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev ** 0.5 * beta_t) / (1-alpha_bar)) * x0 +  ((alpha_t**0.5 *(1-alpha_prod_t_prev)) / (1- alpha_bar))*xt

def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas

    # Get dimensions from x0 to handle both SD v1.4 (64x64) and SDXL (128x128) latents
    batch_size, channels, height, width = x0.shape

    variance_noise_shape = (
            num_inference_steps,
            channels,
            height,
            width)

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xts = torch.zeros((num_inference_steps+1, channels, height, width), device=x0.device, dtype=x0.dtype)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps-t_to_idx[int(t)]
        xts[idx] = x0 * (alpha_bar[t] ** 0.5) +  torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]


    return xts


def encode_text(model, prompts):
    """Encode text prompts. Returns text_embeddings for compatibility."""
    from diffusers import StableDiffusionXLPipeline

    if isinstance(model, StableDiffusionXLPipeline):
        # SDXL: Use encode_prompt to handle dual text encoders
        # Handle both single prompt (string) and multiple prompts (list)
        if isinstance(prompts, list):
            # Encode each prompt separately and concatenate
            all_prompt_embeds = []
            all_pooled_embeds = []
            for prompt in prompts:
                with torch.no_grad():
                    prompt_embeds, _, pooled_embeds, _ = model.encode_prompt(
                        prompt=prompt,
                        prompt_2=None,
                        device=model.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False
                    )
                all_prompt_embeds.append(prompt_embeds)
                all_pooled_embeds.append(pooled_embeds)

            prompt_embeds = torch.cat(all_prompt_embeds, dim=0)
            pooled_embeds = torch.cat(all_pooled_embeds, dim=0)
        else:
            # Single string prompt
            with torch.no_grad():
                prompt_embeds, _, pooled_embeds, _ = model.encode_prompt(
                    prompt=prompts,
                    prompt_2=None,
                    device=model.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )

        # Store pooled embeddings in model for later use
        if not hasattr(model, '_cached_pooled_embeds'):
            model._cached_pooled_embeds = {}
        model._cached_pooled_embeds[prompts if isinstance(prompts, str) else str(prompts)] = pooled_embeds
        return prompt_embeds
    else:
        # SD 1.4: Original implementation
        text_input = model.tokenizer(
            prompts,
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
        return text_encoding

def get_added_cond_kwargs(model, prompt):
    """Get added_cond_kwargs for SDXL (pooled embeddings + time_ids)."""
    from diffusers import StableDiffusionXLPipeline
    import torch

    if isinstance(model, StableDiffusionXLPipeline):
        # Get cached pooled embeddings or encode now
        cache_key = prompt if isinstance(prompt, str) else str(prompt)
        if hasattr(model, '_cached_pooled_embeds') and cache_key in model._cached_pooled_embeds:
            pooled_embeds = model._cached_pooled_embeds[cache_key]
        else:
            # Encode to get pooled embeddings - handle both single and list of prompts
            if isinstance(prompt, list):
                # Encode each prompt separately and concatenate
                all_pooled_embeds = []
                for p in prompt:
                    _, _, pooled, _ = model.encode_prompt(
                        prompt=p,
                        prompt_2=None,
                        device=model.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False
                    )
                    all_pooled_embeds.append(pooled)
                pooled_embeds = torch.cat(all_pooled_embeds, dim=0)
            else:
                _, _, pooled_embeds, _ = model.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    device=model.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
                )

        # Get batch size from pooled embeddings
        batch_size = pooled_embeds.shape[0]

        # Use pooled_embeds dtype to match what DirectInversion does (critical for numerical stability)
        embeds_dtype = pooled_embeds.dtype

        # Create added_cond_kwargs with matching dtype and batch size
        return {
            "text_embeds": pooled_embeds,
            "time_ids": torch.tensor([[512, 512, 0, 0, 512, 512]] * batch_size, device=model.device, dtype=embeds_dtype)
        }
    else:
        return None

def forward_step(model, model_output, timestep, sample):
    next_timestep = min(model.scheduler.config.num_train_timesteps - 2,
                        timestep + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps)

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(pred_original_sample,
                                    model_output,
                                    torch.LongTensor([next_timestep]))
    return next_sample


def get_variance(model, timestep): #, prev_timestep):
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance

def inversion_forward_process(model, x0, 
                            etas = None,    
                            prog_bar = False,
                            prompt = "",
                            cfg_scale = 3.5,
                            num_inference_steps=50, eps = None):

    if not prompt=="":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")

    # Compute added_cond_kwargs once before the loop and reuse it
    # This prevents SDXL from "hallucinating" different cropping parameters at each step
    uncond_added_kwargs = get_added_cond_kwargs(model, "")
    cond_added_kwargs = get_added_cond_kwargs(model, prompt) if not prompt=="" else None

    timesteps = model.scheduler.timesteps.to(model.device)

    # Get dimensions from x0 to handle both SD v1.4 (64x64) and SDXL (128x128) latents
    batch_size, channels, height, width = x0.shape
    variance_noise_shape = (
        num_inference_steps,
        channels,
        height,
        width)

    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device, dtype=x0.dtype)
    t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
    xt = x0
    op = timesteps

    for t in op:
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps-t_to_idx[int(t)]-1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx+1][None]
            # xt = xts_cycle[idx+1][None]

        with torch.no_grad():
            if uncond_added_kwargs is not None:
                out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding, added_cond_kwargs=uncond_added_kwargs)
            else:
                out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)

            if not prompt=="":
                if cond_added_kwargs is not None:
                    cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings, added_cond_kwargs=cond_added_kwargs)
                else:
                    cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)

        if not prompt=="":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample
        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else: 
            # xtm1 =  xts[idx+1][None]
            xtm1 =  xts[idx][None]
            # pred of x0
            pred_original_sample = (xt - (1-alpha_bar[t])  ** 0.5 * noise_pred ) / alpha_bar[t] ** 0.5
            
            # direction to xt
            prev_timestep = t - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
            alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
            
            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance ) ** (0.5) * noise_pred

            mu_xt = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

            z = (xtm1 - mu_xt ) / ( etas[idx] * variance ** 0.5 )
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + ( etas[idx] * variance ** 0.5 )*z
            xts[idx] = xtm1

    if not zs is None: 
        zs[0] = torch.zeros_like(zs[0]) 

    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta = 0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep) #, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (0.5) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z =  eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample

def inversion_reverse_process(model,
                    xT, 
                    etas = 0,
                    prompts = "",
                    cfg_scales = None,
                    prog_bar = False,
                    zs = None,
                    controller=None,
                    asyrp = False):

    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1,1,1,1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    # Compute added_cond_kwargs once before the loop and reuse it
    # This prevents SDXL from "hallucinating" different cropping parameters at each step
    uncond_added_kwargs = get_added_cond_kwargs(model, [""] * batch_size)
    cond_added_kwargs = get_added_cond_kwargs(model, prompts) if prompts else None

    if etas is None: etas = 0
    if type(etas) in [int, float]: etas = [etas]*model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = timesteps[-zs.shape[0]:]

    t_to_idx = {int(v):k for k,v in enumerate(timesteps[-zs.shape[0]:])}

    for t in op:
        idx = model.scheduler.num_inference_steps-t_to_idx[int(t)]-(model.scheduler.num_inference_steps-zs.shape[0]+1)

        ## Unconditional embedding
        with torch.no_grad():
            if uncond_added_kwargs is not None:
                uncond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding, added_cond_kwargs=uncond_added_kwargs)
            else:
                uncond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=uncond_embedding)

        ## Conditional embedding
        if prompts:
            with torch.no_grad():
                if cond_added_kwargs is not None:
                    cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings, added_cond_kwargs=cond_added_kwargs)
                else:
                    cond_out = model.unet.forward(xt, timestep=t, encoder_hidden_states=text_embeddings)
            
        
        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (cond_out.sample - uncond_out.sample)
        else: 
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1  
        xt = reverse_step(model, noise_pred, t, xt, eta = etas[idx], variance_noise = z) 
        if controller is not None:
            xt = controller.step_callback(xt)        
    return xt, zs


