from brew.envs.desktop_env.providers.base import VMManager, Provider


def create_vm_manager_and_provider(provider_name: str, region: str, use_proxy: bool = False):
    """
    Factory function to get the Virtual Machine Manager and Provider instances based on the provided provider name.
    
    Args:
        provider_name (str): The name of the provider (e.g., "aws", "vmware", etc.)
        region (str): The region for the provider
        use_proxy (bool): Whether to use proxy-enabled providers (currently only supported for AWS)
    """
    provider_name = provider_name.lower().strip()
    if provider_name == "vmware":
        from brew.envs.desktop_env.providers.vmware.manager import VMwareVMManager
        from brew.envs.desktop_env.providers.vmware.provider import VMwareProvider
        return VMwareVMManager(), VMwareProvider(region)
    elif provider_name == "virtualbox":
        from brew.envs.desktop_env.providers.virtualbox.manager import VirtualBoxVMManager
        from brew.envs.desktop_env.providers.virtualbox.provider import VirtualBoxProvider
        return VirtualBoxVMManager(), VirtualBoxProvider(region)
    elif provider_name in ["aws", "amazon web services"]:
        from brew.envs.desktop_env.providers.aws.manager import AWSVMManager
        from brew.envs.desktop_env.providers.aws.provider import AWSProvider
        return AWSVMManager(), AWSProvider(region)
    elif provider_name == "azure":
        from brew.envs.desktop_env.providers.azure.manager import AzureVMManager
        from brew.envs.desktop_env.providers.azure.provider import AzureProvider
        return AzureVMManager(), AzureProvider(region)
    elif provider_name == "docker":
        from brew.envs.desktop_env.providers.docker.manager import DockerVMManager
        from brew.envs.desktop_env.providers.docker.provider import DockerProvider
        return DockerVMManager(), DockerProvider(region)
    elif provider_name == "aliyun":
        from brew.envs.desktop_env.providers.aliyun.manager import AliyunVMManager
        from brew.envs.desktop_env.providers.aliyun.provider import AliyunProvider
        return AliyunVMManager(), AliyunProvider()
    elif provider_name == "volcengine":
        from brew.envs.desktop_env.providers.volcengine.manager import VolcengineVMManager
        from brew.envs.desktop_env.providers.volcengine.provider import VolcengineProvider
        return VolcengineVMManager(), VolcengineProvider()
    else:
        raise NotImplementedError(f"{provider_name} not implemented!")
