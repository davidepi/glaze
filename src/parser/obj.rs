use crate::parser::parser::GeometryParser;
use crate::parser::{GeometryError, ParseGeometryError, ParsedGeometry};
use std::convert::TryInto;
use std::fmt::Debug;
use std::str::{FromStr, SplitWhitespace};

const MAX_I32: &str = "2147483647";

const FACE_2VERT: &str = "face should have at least 3 vertices";
const PARSE_FAIL: &str = "failed to parse float";
const MISSING_VALUE: &str = "missing value";

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Obj {
    file: String,
}

impl Obj {
    pub fn new(file: String) -> Obj {
        Obj { file }
    }
}

impl GeometryParser for Obj {
    fn parse(&self) -> Result<Vec<ParsedGeometry>, GeometryError> {
        let mut res = Vec::new();
        let content = std::fs::read_to_string(&self.file)?;
        let split = content.split('\n');
        let mut name = String::new();
        let mut vv = Vec::new();
        let mut vt = Vec::new();
        let mut vn = Vec::new();
        let mut ff = Vec::new();
        for line in split.into_iter().enumerate() {
            let mut iterator = line.1.split_whitespace();
            if let Err(err) = match iterator.next() {
                Some(_letter @ "v") => parse_float_line(&mut iterator, true, &mut vv),
                Some(_letter @ "vn") => parse_float_line(&mut iterator, true, &mut vn),
                Some(_letter @ "vt") => parse_float_line(&mut iterator, false, &mut vt),
                Some(_letter @ "f") => parse_face_line(&mut iterator, &mut ff),
                Some(_letter @ "o") => {
                    // if v is filled create the 3d object, then clear all the data and start anew
                    if !vv.is_empty() {
                        res.push(ParsedGeometry {
                            name,
                            vv,
                            vn,
                            vt: vt.into_iter().map(|x| [x[0], x[1]]).collect::<Vec<_>>(),
                            ff,
                        });
                    }
                    name = iterator.next().unwrap_or("").to_string();
                    vv = Vec::new();
                    vt = Vec::new();
                    vn = Vec::new();
                    ff = Vec::new();
                    Ok(())
                }
                _ => {
                    Ok(()) /* skip unrecognized (probably comments) */
                }
            } {
                return Err(GeometryError::from(ParseGeometryError {
                    file: self.file.clone(),
                    line: line.0 + 1,
                    cause: err,
                }));
            }
        }
        // use the remainder data to fill the last object before returning
        if !vv.is_empty() {
            res.push(ParsedGeometry {
                name,
                vv,
                vn,
                vt: vt.into_iter().map(|x| [x[0], x[1]]).collect::<Vec<_>>(),
                ff,
            });
        }
        Ok(res)
    }
}

fn parse_float_line(
    split: &mut SplitWhitespace,
    args3: bool,
    container: &mut Vec<[f32; 3]>,
) -> Result<(), String> {
    let xyz = (split.next(), split.next(), split.next());
    let parsed = match xyz {
        (Some(x), Some(y), None) => Ok((f32::from_str(x), f32::from_str(y), f32::from_str("NaN"))),
        (Some(x), Some(y), Some(z)) => Ok((f32::from_str(x), f32::from_str(y), f32::from_str(z))),
        _ => Err(MISSING_VALUE.to_owned()),
    }?;
    match parsed {
        (Ok(xx), Ok(yy), Ok(zz)) => {
            if args3 && f32::is_nan(zz) {
                Err(MISSING_VALUE.to_owned())
            } else {
                container.push([xx, yy, zz]);
                Ok(())
            }
        }
        _ => Err(PARSE_FAIL.to_owned()),
    }
}

fn parse_face_line(
    split: &mut SplitWhitespace,
    container: &mut Vec<[i32; 9]>,
) -> Result<(), String> {
    let faces = triangulate(split.collect::<Vec<_>>())?;
    container.extend(faces);
    Ok(())
}

fn triangulate(ngon: Vec<&str>) -> Result<Vec<[i32; 9]>, String> {
    if ngon.len() < 3 {
        Err(FACE_2VERT.to_string())
    } else {
        // triangulate if necessary
        let triangles = (2..)
            .take(ngon.len() - 2)
            .map(|x| vec![ngon[0], ngon[x - 1], ngon[x]])
            .collect::<Vec<_>>();
        // convert v/vn/vt into [v0,vt0,vn0,v1,vt1,vn1,v2,vt2,vn2]
        let mut res = Vec::new();
        for tris in triangles {
            let mut face = Vec::new();
            for val in tris {
                let split = val
                    .split('/')
                    .chain(vec![MAX_I32; 3])
                    .take(3)
                    .map(|x| if x.is_empty() { MAX_I32 } else { x })
                    .map(i32::from_str)
                    .collect::<Result<Vec<_>, _>>();
                match split {
                    Ok(value) => face.extend(value.into_iter()),
                    Err(e) => return Err(e.to_string()),
                }
            }
            res.push(face.try_into().unwrap());
        }
        Ok(res)
    }
}

// OBJ TO STRING
// TODO: maybe this can be reused in the Mesh class, it is nicer than current implementation
// fn to_string(&self) -> String {
//     let p_str = self
//         .v
//         .iter()
//         .map(|x| to_obj_line(x, "v ", 3))
//         .collect::<Vec<_>>()
//         .join("\n");
//     let n_str = self
//         .n
//         .iter()
//         .map(|x| to_obj_line(x, "vn ", 3))
//         .collect::<Vec<_>>()
//         .join("\n");
//     let t_str = self
//         .t
//         .iter()
//         .map(|x| to_obj_line(x, "vt ", 2))
//         .collect::<Vec<_>>()
//         .join("\n");
//     [self.header.clone(), p_str, n_str, t_str].join("\n")
// }
//
// fn to_obj_line<T: Index<u8>>(p: &T, begin: &str, values: usize) -> String
//     where
//         <T as Index<u8>>::Output: ToString,
// {
//     let string = (0 as u8..)
//         .take(values)
//         .map(|x| p[x].to_string())
//         .collect::<Vec<_>>()
//         .join(" ");
//     [String::from(begin), string].join("\n")
// }

#[cfg(test)]
mod tests {
    use crate::parser::obj::FACE_2VERT;
    use crate::parser::parser::GeometryParser;
    use crate::parser::{GeometryError, Obj};
    use std::path::PathBuf;

    // robust enough for tests
    fn get_resource(filename: &'static str) -> String {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("resources/test/obj");
        path.push(filename);
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn obj_two_vertices_face() {
        let filename = get_resource("2vertface.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_err(), "Expected error, nothing was raised");
        let err = parsed.err().unwrap();
        match &err {
            GeometryError::IoError(_) => {
                assert!(false, "wrong error type")
            }
            GeometryError::ParseError(err) => {
                assert_eq!(err.file, filename);
                assert_eq!(err.line, 3);
                assert_eq!(err.cause, FACE_2VERT);
            }
        }
    }

    #[test]
    fn obj_no_name() {
        let filename = get_resource("noname.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_ok());
        let res = parsed.unwrap();
        assert_eq!(res[0].name, "");
    }

    #[test]
    fn obj_float_in_face() {
        let filename = get_resource("float_in_face.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_err(), "Expected error, nothing was raised");
        let err = parsed.err().unwrap();
        match &err {
            GeometryError::IoError(_) => {
                assert!(false, "wrong error type")
            }
            GeometryError::ParseError(err) => {
                assert_eq!(err.file, filename);
                assert_eq!(err.line, 19);
                assert_eq!(err.cause, "invalid digit found in string")
            }
        }
    }

    #[test]
    fn obj_multispace() {
        let filename = get_resource("multispace.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_ok());
        let res = parsed.unwrap();
        assert_eq!(res[0].vv.len(), 3);
        assert_eq!(res[0].vv[0], [0.0, 0.0, 0.0]);
        assert_eq!(res[0].vv[1], [1.0, 0.0, 0.0]);
        assert_eq!(res[0].vv[2], [1.0, 1.0, 0.0]);
    }

    #[test]
    fn obj_triangulate() {
        let filename = get_resource("ngon.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_ok());
        let res = parsed.unwrap();
        assert_eq!(res[0].ff.len(), 14);
    }

    #[test]
    fn obj_negative_tris_index() {
        let filename = get_resource("neg_vertices.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_ok());
        let res = parsed.unwrap();
        assert_eq!(res[0].ff.len(), 1);
        assert_eq!(res[0].ff[0][0], -1);
        assert_eq!(res[0].ff[0][1], -1);
        assert_eq!(res[0].ff[0][2], -6);
    }

    #[test]
    fn obj_vertex_parse() {
        let filename = get_resource("pyramid.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_ok());
        let res = parsed.unwrap();
        assert_eq!(res[0].name, "SquarePyr");
        assert_eq!(res[0].vv[0], [0.0, 0.0, 0.0]);
        assert_eq!(res[0].vv[1], [1.0, 0.0, 0.0]);
        assert_eq!(res[0].vv[2], [1.0, 1.0, 0.0]);
        assert_eq!(res[0].vv[3], [0.0, 1.0, 0.0]);
        assert_eq!(res[0].vv[4], [0.5, 0.5, 1.0]);
        assert_eq!(res[0].vn.len(), 6);
        assert_eq!(res[0].ff.len(), 6);
    }

    #[test]
    fn obj_missing_vt() {
        let filename = get_resource("pyramid.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_ok());
        let res = parsed.unwrap();
        assert_eq!(res[0].name, "SquarePyr");
        assert!(res[0].vt.is_empty());
        assert_eq!(
            res[0].ff[0],
            [5, 2147483647, 1, 2, 2147483647, 1, 3, 2147483647, 1]
        );
        assert_eq!(
            res[0].ff[1],
            [4, 2147483647, 2, 5, 2147483647, 2, 3, 2147483647, 2]
        );
        assert_eq!(
            res[0].ff[2],
            [1, 2147483647, 3, 3, 2147483647, 3, 2, 2147483647, 3]
        );
        assert_eq!(
            res[0].ff[3],
            [5, 2147483647, 4, 1, 2147483647, 4, 2, 2147483647, 4]
        );
        assert_eq!(
            res[0].ff[4],
            [4, 2147483647, 5, 1, 2147483647, 5, 5, 2147483647, 5]
        );
        assert_eq!(
            res[0].ff[5],
            [1, 2147483647, 6, 4, 2147483647, 6, 3, 2147483647, 6]
        );
    }
    #[test]
    fn obj_multiple_objects() {
        let filename = get_resource("multi.obj");
        let parsed = Obj::new(filename.clone()).parse();
        assert!(parsed.is_ok());
        let res = parsed.unwrap();
        assert_ne!(res[0].name, res[1].name);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].vv.len(), 8);
        assert_eq!(res[0].vt.len(), 0);
        assert_eq!(res[0].vn.len(), 6);
        assert_eq!(res[0].ff.len(), 12);
        assert_eq!(res[0].vv.len(), 8);
        assert_eq!(res[0].vt.len(), 0);
        assert_eq!(res[0].vn.len(), 6);
        assert_eq!(res[0].ff.len(), 12);
    }
}
