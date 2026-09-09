"""Allow ``python -m thermotwin`` to invoke the supported CLI."""

from thermotwin.cli import main

raise SystemExit(main())

