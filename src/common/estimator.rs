use std::process::Command;
use std::io;

#[derive(Debug, Clone, Copy)]
pub enum Norm {
    L2,
    Infinity,
}

impl Norm {
    pub fn as_str(&self) -> &str {
        match self {
            Norm::L2 => "2",
            Norm::Infinity => "oo",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SISParameters {
    pub n: u32,
    pub m: u32,
    pub q: u32,
    pub length_bound: u32,
    pub norm: Norm,
}

#[derive(Debug)]
pub struct EstimatorResult {
    pub secpar: f64,
}

pub fn estimate_sis_security(params: &SISParameters) -> Result<EstimatorResult, io::Error> {
    let script_path = std::env::current_dir()?.join("run_sage_estimator.sh");
    
    let output = Command::new("bash")
        .arg(script_path)
        .arg(params.n.to_string())
        .arg(params.m.to_string())
        .arg(params.q.to_string())
        .arg(params.length_bound.to_string())
        .arg(params.norm.as_str())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Estimator script failed: {}", stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let secpar: f64 = stdout
        .trim()
        .parse()
        .map_err(|e| io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Failed to parse output: {}", e)
        ))?;

    Ok(EstimatorResult { secpar })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator() {
        let params = SISParameters {
            n: 113,
            m: 1000,
            q: 2048,
            length_bound: 512,
            norm: Norm::L2,
        };

        let result = estimate_sis_security(&params).unwrap();
        println!("Log2(ROP): {}", result.secpar);
        assert!(result.secpar > 0.0);
    }
}
