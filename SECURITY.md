# Security Policy

## Reporting a Vulnerability

1. **Do NOT** open a public issue
2. Email the maintainer directly
3. Include: description, steps to reproduce, potential impact

## Security Measures

- API keys stored in environment variables
- Input validation via Pydantic models
- CORS restricted in production
- No PII stored in vector indices
