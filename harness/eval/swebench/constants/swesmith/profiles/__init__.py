from .base import RepoProfile, registry

# Auto-import all profile modules to populate the registry
from . import c
from . import cpp
from . import csharp
from . import java
from . import javascript
from . import php
from . import typescript
from . import python
from . import golang
from . import rust

__all__ = ["RepoProfile", "registry"]
