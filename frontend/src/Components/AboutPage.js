import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography'
import { Avatar } from '@mui/material';
import { Card, CardActions, CardContent, Button } from '@mui/material';
import { Container, Stack, Item } from '@mui/material';
import Pratyanshu from "../assets/Pratyanshu.jpeg"
import ProfChiranjeevi from "../assets/ProfChiranjeevi.jpeg"
import Sai from "../assets/Sai.jpeg"
import Snehal from "../assets/Snehal.jpeg"
import Utkarsh from "../assets/Utkarsh.jpeg"

const contributors = [
  {
    name: "Chiranjeevi Yarra",
    role: "Advisor",
    image: ProfChiranjeevi
  },
  {
    name: "Sai Nanduri",
    role: "Developer",
    image: Sai
  },
  {
    name: "Snehal Ranjan",
    role: "Developer",
    image: Snehal
  },
  {
    name: "Pratyanshu Pandey",
    role: "Developer",
    image: Pratyanshu
  },
  {
    name: "Utkarsh Upadhyay",
    role: "Developer",
    image: Utkarsh
  }
]

const AboutPage = () => {
  return (
    <Box sx={{
      flexGrow: 1,
      backgroundColor: 'background.default',
      paddingBottom: 10
    }}>
      <Grid container columnSpacing={2} rowSpacing={3} minHeight={600}>
        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
          <Box sx={{
            flexGrow: 1,
            backgroundColor: 'background.paper',
            padding: 10
          }}>
            <Typography
              variant='h2'
              // noWrap
              component="div"
              paddingBottom={3}>
                Project Description
            </Typography>
            <Typography
              variant='body1'
              marginLeft={2}
              marginRight={2}
              // noWrap
              component="div">
              The project aims to identify the intent behind spoken english sentences. 
              The system first records the input, then translates it to text using state of 
              the art speech to text translation mechanisms. The user is then provided and 
              option to correct the sentences if required. Then the model is run on the text to 
              output the intent behind it.
            </Typography>
          </Box>
        </Grid>

        <Grid item xs={12} sm={12} md={12} lg={12} xl={12}>
        <Typography
              variant='h2'
              // noWrap
              component="div"
              paddingBottom={3}>
                Contributors
            </Typography>
        </Grid>

        {/* <Grid item xs={1} sm={1} md={1} lg={1} xl={1}></Grid> */}
        
        {contributors.map((ele) =>
          <Grid item xs={12} sm={12} md={6} lg={2.4} xl={2.4} sx={{ alignContent: 'center', justifyContent: 'center' }}>
            <Card sx={{ maxWidth: 200 , marginLeft: 5, marginRight: 5}}>
              <CardContent>
                <Stack>
                  <Avatar
                    alt={ele.name}
                    src={ele.image}
                    sx={{ width: 56, height: 56, alignSelf: 'center' }}
                  />
                  <Typography sx={{ fontSize: 14 }} color="text.secondary" marginTop={1} gutterBottom>
                    {ele.name}
                  </Typography>
                  <Typography sx={{ fontSize: 14 }} color="text.secondary" gutterBottom>
                    {ele.role}
                  </Typography>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* <Grid item xs={1} sm={1} md={1} lg={1} xl={1}></Grid> */}

      </Grid>
    </Box>
  )
}

export default AboutPage